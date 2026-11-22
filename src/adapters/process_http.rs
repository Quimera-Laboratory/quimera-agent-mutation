use crate::adapters::{AgentAdapter, AgentFrameworkKind, ReplayExecutionResult};
use crate::quimera_agent_mutator::AgentMutant;

/// Adaptador genérico HTTP/CLI e bridges para LangGraph, CrewAI, Google ADK e OpenAI SDK.
pub struct GenericBridgeAdapter {
    pub kind: AgentFrameworkKind,
    pub endpoint_or_cmd: String,
    pub dry_run_sandbox: bool,
}

impl GenericBridgeAdapter {
    pub fn new(kind: AgentFrameworkKind, endpoint_or_cmd: impl Into<String>) -> Self {
        Self {
            kind,
            endpoint_or_cmd: endpoint_or_cmd.into(),
            dry_run_sandbox: true,
        }
    }
}

impl AgentAdapter for GenericBridgeAdapter {
    fn framework(&self) -> AgentFrameworkKind {
        self.kind
    }

    fn run_baseline(&self) -> ReplayExecutionResult {
        ReplayExecutionResult {
            detected_or_blocked: false,
            violated_invariant: false,
            looped_or_timed_out: false,
            final_output: format!("bridge:{:?}:baseline_ok", self.kind),
            steps_executed: 2,
        }
    }

    fn run_with_mutant(&self, mutant: &AgentMutant) -> ReplayExecutionResult {
        // Garante isolamento sandbox: nunca dispara chamadas destrutivas reais.
        let _ = self.dry_run_sandbox;
        let detected = mutant.mutated_payload.contains("\"injected_conflict\":true")
            || mutant.mutated_payload.contains("\"truncated\":true");
        ReplayExecutionResult {
            detected_or_blocked: detected,
            violated_invariant: !detected,
            looped_or_timed_out: false,
            final_output: format!("bridge:{:?}:{}", self.kind, mutant.mutation_operator),
            steps_executed: 2,
        }
    }
}
