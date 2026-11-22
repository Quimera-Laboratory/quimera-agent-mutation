use crate::adapters::{AgentAdapter, ReplayExecutionResult};
use crate::llm_router::BudgetGuard;
use crate::quimera_agent_mutator::AgentMutant;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct SnapshotReplayOutcome {
    pub mutant: AgentMutant,
    pub runs: Vec<ReplayExecutionResult>,
}

/// Executor paralelo de Branching Snapshot Replay em memória (CoW lógico em t_0).
pub struct SandboxReplayRunner {
    pub samples_per_mutant: usize,
    pub step_timeout_limit: usize,
}

impl SandboxReplayRunner {
    pub fn new(samples_per_mutant: usize, step_timeout_limit: usize) -> Self {
        Self {
            samples_per_mutant: samples_per_mutant.max(1),
            step_timeout_limit,
        }
    }

    pub fn execute_campaign(
        &self,
        adapter: &dyn AgentAdapter,
        mutants: &[AgentMutant],
        budget: &BudgetGuard,
    ) -> Result<Vec<SnapshotReplayOutcome>, String> {
        // 1. Verificação obrigatória de baseline estável
        for _ in 0..self.samples_per_mutant {
            let base = adapter.run_baseline();
            if base.violated_invariant || base.looped_or_timed_out {
                return Err("UNSTABLE_BASELINE: baseline agent failed evals or timed out".to_string());
            }
        }

        // 2. Respeita teto de mutantes e executa em paralelo via rayon
        let limit = mutants.len().min(budget.max_mutants);
        let selected = &mutants[..limit];

        let outcomes: Vec<SnapshotReplayOutcome> = selected
            .par_iter()
            .filter_map(|mutant| {
                // Custo nominal de replay por mutante (128 tokens simulados no orçamento)
                if !budget.try_consume(128, 10) {
                    return None;
                }
                let mut runs = Vec::with_capacity(self.samples_per_mutant);
                for _ in 0..self.samples_per_mutant {
                    let mut res = adapter.run_with_mutant(mutant);
                    if res.steps_executed > self.step_timeout_limit {
                        res.looped_or_timed_out = true;
                    }
                    runs.push(res);
                }
                Some(SnapshotReplayOutcome {
                    mutant: mutant.clone(),
                    runs,
                })
            })
            .collect();

        Ok(outcomes)
    }
}
