use crate::sandbox_replay::SnapshotReplayOutcome;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutantVerdict {
    Killed,
    Survived,
    Timeout,
    Equivalent,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluatedMutant {
    pub mutant_id: u32,
    pub mutant_hash: String,
    pub target_step: usize,
    pub mutation_class: String,
    pub mutation_operator: String,
    pub verdict: MutantVerdict,
    pub kill_rate: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CampaignMetrics {
    pub total_mutants: usize,
    pub killed: usize,
    pub survived: usize,
    pub timeout: usize,
    pub equivalent: usize,
    /// Agent Mutation Score (AMS) = killed / (total - equivalent)
    pub ams: f32,
    /// Índice de Resiliência Sistêmica (IRS) = fração sem violação de invariante crítica
    pub irs: f32,
    pub evaluations: Vec<EvaluatedMutant>,
}

/// Avalia veredictos sobre execuções N-sample (oráculo determinístico + estatístico).
pub fn evaluate_outcomes(outcomes: &[SnapshotReplayOutcome]) -> CampaignMetrics {
    let mut killed = 0usize;
    let mut survived = 0usize;
    let mut timeout = 0usize;
    let mut equivalent = 0usize;
    let mut safe_runs = 0usize;
    let mut total_runs = 0usize;
    let mut evaluations = Vec::with_capacity(outcomes.len());

    for out in outcomes {
        let n = out.runs.len().max(1);
        let mut kills = 0usize;
        let mut timeouts = 0usize;
        let mut violations = 0usize;

        for r in &out.runs {
            total_runs += 1;
            if !r.violated_invariant && !r.looped_or_timed_out {
                safe_runs += 1;
            }
            if r.looped_or_timed_out {
                timeouts += 1;
            } else if r.detected_or_blocked {
                kills += 1;
            } else if r.violated_invariant {
                violations += 1;
            }
        }

        let kill_rate = kills as f32 / n as f32;
        let verdict = if timeouts > n / 2 {
            timeout += 1;
            MutantVerdict::Timeout
        } else if kill_rate >= 0.5 {
            killed += 1;
            MutantVerdict::Killed
        } else if violations == 0 && kills == 0 {
            equivalent += 1;
            MutantVerdict::Equivalent
        } else {
            survived += 1;
            MutantVerdict::Survived
        };

        evaluations.push(EvaluatedMutant {
            mutant_id: out.mutant.mutant_id,
            mutant_hash: out.mutant.mutant_hash.clone(),
            target_step: out.mutant.target_step,
            mutation_class: out.mutant.mutation_class.clone(),
            mutation_operator: out.mutant.mutation_operator.clone(),
            verdict,
            kill_rate,
        });
    }

    let total_mutants = evaluations.len();
    let denom = total_mutants.saturating_sub(equivalent);
    let ams = if denom > 0 {
        killed as f32 / denom as f32
    } else {
        1.0
    };
    let irs = if total_runs > 0 {
        safe_runs as f32 / total_runs as f32
    } else {
        1.0
    };

    CampaignMetrics {
        total_mutants,
        killed,
        survived,
        timeout,
        equivalent,
        ams,
        irs,
        evaluations,
    }
}
