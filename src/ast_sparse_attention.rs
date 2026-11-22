/// Sparse attention mask guided by the Agent's execution DAG + sliding window.
#[derive(Clone, Debug)]
pub struct AstSparseAttentionMask {
    pub seq_len: usize,
    pub window: usize,
    /// Adjacency list of allowed attention targets per step index.
    pub allowed_targets: Vec<Vec<usize>>,
}

impl AstSparseAttentionMask {
    /// Builds a causal + DAG-guided sparse attention mask for a trajectory of length `seq_len`.
    pub fn build(seq_len: usize, window: usize, dag_edges: &[(usize, usize)]) -> Self {
        let mut allowed_targets = vec![Vec::new(); seq_len];

        for i in 0..seq_len {
            let start = i.saturating_sub(window);
            for j in start..=i {
                allowed_targets[i].push(j);
            }
        }

        for &(src, dst) in dag_edges {
            if src < seq_len && dst < seq_len {
                if !allowed_targets[dst].contains(&src) {
                    allowed_targets[dst].push(src);
                }
                if !allowed_targets[src].contains(&dst) {
                    allowed_targets[src].push(dst);
                }
            }
        }

        for targets in &mut allowed_targets {
            targets.sort_unstable();
            targets.dedup();
        }

        Self {
            seq_len,
            window,
            allowed_targets,
        }
    }

    #[inline]
    pub fn allows(&self, query_idx: usize, key_idx: usize) -> bool {
        if query_idx >= self.seq_len || key_idx >= self.seq_len {
            return false;
        }
        self.allowed_targets[query_idx].binary_search(&key_idx).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_includes_sliding_window_and_dag_edges() {
        let mask = AstSparseAttentionMask::build(6, 1, &[(0, 5), (2, 4)]);
        assert!(mask.allows(5, 5));
        assert!(mask.allows(5, 4));
        assert!(mask.allows(5, 0));
        assert!(!mask.allows(5, 2));
        assert!(mask.allows(4, 2));
    }
}
