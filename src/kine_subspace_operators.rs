/// Q16.16 fixed-point scale factor.
const Q16_SCALE: f32 = 65536.0;

/// Orthogonal Givens-block rotator in SO(128) via Cayley-style 2x2 planar rotations.
#[derive(Clone, Debug)]
pub struct PrecomputedCayley128 {
    /// (cos_theta, sin_theta) for 64 disjoint 2D planes covering 128 dims.
    planes: [(f32, f32); 64],
}

impl PrecomputedCayley128 {
    pub fn new_identity() -> Self {
        Self {
            planes: [(1.0, 0.0); 64],
        }
    }

    /// Creates a deterministic orthogonal SO(128) rotation from a seed angle.
    pub fn from_seed_angle(base_angle: f32) -> Self {
        let mut planes = [(1.0, 0.0); 64];
        for (k, slot) in planes.iter_mut().enumerate() {
            let theta = base_angle * ((k as f32 + 1.0) / 64.0);
            *slot = (theta.cos(), theta.sin());
        }
        Self { planes }
    }

    /// Applies the orthogonal rotation in-place preserving ||z||_2.
    pub fn rotate_in_place(&self, z: &mut [f32; 128]) {
        for (k, &(c, s)) in self.planes.iter().enumerate() {
            let i = 2 * k;
            let x = z[i];
            let y = z[i + 1];
            z[i] = c * x - s * y;
            z[i + 1] = s * x + c * y;
        }
    }
}

/// Dual Caputo Fractional Memory Engine in Q16.16 fixed-point arithmetic.
/// Tracks power-law context drift over forward trajectory turns and backward layers.
#[derive(Clone, Debug)]
pub struct DualCaputoQ16Engine {
    pub alpha_fwd: f32,
    pub mem_fwd: usize,
    pub damp_fwd: f32,
    pub alpha_bwd: f32,
    pub mem_bwd: usize,
    pub damp_bwd: f32,
    weights_fwd_q16: Vec<i32>,
    weights_bwd_q16: Vec<i32>,
}

impl DualCaputoQ16Engine {
    pub fn new(
        alpha_fwd: f32,
        mem_fwd: usize,
        damp_fwd: f32,
        alpha_bwd: f32,
        mem_bwd: usize,
        damp_bwd: f32,
    ) -> Self {
        let weights_fwd_q16 = Self::compute_gl_weights_q16(alpha_fwd, mem_fwd);
        let weights_bwd_q16 = Self::compute_gl_weights_q16(alpha_bwd, mem_bwd);
        Self {
            alpha_fwd,
            mem_fwd,
            damp_fwd,
            alpha_bwd,
            mem_bwd,
            damp_bwd,
            weights_fwd_q16,
            weights_bwd_q16,
        }
    }

    fn compute_gl_weights_q16(alpha: f32, horizon: usize) -> Vec<i32> {
        let mut weights = Vec::with_capacity(horizon);
        let mut w = 1.0_f32;
        for j in 1..=horizon {
            w *= (j as f32 - 1.0 - alpha) / (j as f32);
            let mag = w.abs().min(1.0);
            weights.push((mag * Q16_SCALE).round() as i32);
        }
        weights
    }

    /// Applies dual fractional memory accumulation at `pos`.
    pub fn apply_dual_caputo(
        &self,
        pos: usize,
        seq_states: &[Vec<f32>],
        backward_states: &[Vec<f32>],
        out: &mut [f32],
    ) {
        let dim = out.len();
        if dim == 0 || seq_states.is_empty() {
            return;
        }

        // Forward power-law memory over prior agent turns
        for (lag_idx, &w_q16) in self.weights_fwd_q16.iter().enumerate() {
            let lag = lag_idx + 1;
            if pos < lag {
                break;
            }
            let prev = &seq_states[pos - lag];
            let scale = (w_q16 as f32 / Q16_SCALE) * self.damp_fwd;
            for d in 0..dim.min(prev.len()) {
                out[d] += scale * prev[d];
            }
        }

        // Backward fractional coupling (if bidirectional states are provided)
        if !backward_states.is_empty() {
            for (lag_idx, &w_q16) in self.weights_bwd_q16.iter().enumerate() {
                let lag = lag_idx + 1;
                if pos + lag >= backward_states.len() {
                    break;
                }
                let next = &backward_states[pos + lag];
                let scale = (w_q16 as f32 / Q16_SCALE) * self.damp_bwd;
                for d in 0..dim.min(next.len()) {
                    out[d] += scale * next[d];
                }
            }
        }
    }
}

/// Sparse Neumann-2 Sheaf Diffuser ($\Delta_{\mathcal{S}}$) enforcing global trajectory coherence.
#[derive(Clone, Debug)]
pub struct SparseNeumannSheaf {
    pub coupling: f32,
}

impl SparseNeumannSheaf {
    pub fn new(coupling: f32) -> Self {
        Self {
            coupling: coupling.clamp(0.0, 0.49),
        }
    }

    /// Performs 2-step Neumann sheaf Laplacian diffusion over consecutive step states.
    pub fn diffuse_sequence(&self, seq_states: &mut [Vec<f32>]) {
        let n = seq_states.len();
        if n < 2 || self.coupling <= 0.0 {
            return;
        }

        let eta = self.coupling;
        for _pass in 0..2 {
            let snapshot = seq_states.to_vec();
            for i in 0..n {
                let dim = snapshot[i].len();
                for d in 0..dim {
                    let mut laplacian = 0.0_f32;
                    if i > 0 && d < snapshot[i - 1].len() {
                        laplacian += snapshot[i][d] - snapshot[i - 1][d];
                    }
                    if i + 1 < n && d < snapshot[i + 1].len() {
                        laplacian += snapshot[i][d] - snapshot[i + 1][d];
                    }
                    seq_states[i][d] -= eta * laplacian;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cayley_preserves_l2_norm() {
        let rot = PrecomputedCayley128::from_seed_angle(0.75);
        let mut v = [0.0_f32; 128];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = (i as f32 * 0.13).sin();
        }
        let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        rot.rotate_in_place(&mut v);
        let norm_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-4);
    }

    #[test]
    fn caputo_and_sheaf_evolve_states_boundedly() {
        let caputo = DualCaputoQ16Engine::new(0.6, 4, 0.08, 0.5, 2, 0.05);
        let sheaf = SparseNeumannSheaf::new(0.15);
        let mut states = vec![vec![1.0; 128], vec![0.5; 128], vec![0.0; 128]];
        let mut h = states[2].clone();
        caputo.apply_dual_caputo(2, &states, &[], &mut h);
        states[2] = h;
        assert!(states[2][0] > 0.0);
        sheaf.diffuse_sequence(&mut states);
        assert!(states.iter().flatten().all(|v| v.is_finite()));
    }
}
