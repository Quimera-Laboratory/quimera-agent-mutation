use crate::adapters::{AgentAdapter, AgentFrameworkKind, ReplayExecutionResult};
use crate::quimera_agent_mutator::AgentMutant;

/// Adaptador MCP (Model Context Protocol) com interceptação de tool-calling em sandbox.
pub struct McpSandboxAdapter {
    pub server_name: String,
    pub strict_guardrails: bool,
}

impl McpSandboxAdapter {
    pub fn new(server_name: impl Into<String>, strict_guardrails: bool) -> Self {
        Self {
            server_name: server_name.into(),
            strict_guardrails,
        }
    }
}

impl AgentAdapter for McpSandboxAdapter {
    fn framework(&self) -> AgentFrameworkKind {
        AgentFrameworkKind::Mcp
    }

    fn run_baseline(&self) -> ReplayExecutionResult {
        ReplayExecutionResult {
            detected_or_blocked: false,
            violated_invariant: false,
            looped_or_timed_out: false,
            final_output: format!("mcp:{}:ok", self.server_name),
            steps_executed: 3,
        }
    }

    fn run_with_mutant(&self, mutant: &AgentMutant) -> ReplayExecutionResult {
        let p = mutant.mutated_payload.as_str();
        // Simula comportamento real de um agente MCP sob mutação:
        // - Loops se action=continue indefinidamente ou delay bizantino
        if p.contains("\"action\":\"continue\"") && !self.strict_guardrails {
            return ReplayExecutionResult {
                detected_or_blocked: false,
                violated_invariant: true,
                looped_or_timed_out: true,
                final_output: "max_steps_exceeded".to_string(),
                steps_executed: 64,
            };
        }
        // - Bypass de confirmação humana: se strict_guardrails=true, agente bloqueia (KILLED);
        //   se strict_guardrails=false, agente executa ação insegura sem bloquear (SURVIVED).
        if p.contains("\"require_confirmation\":false") {
            return ReplayExecutionResult {
                detected_or_blocked: self.strict_guardrails,
                violated_invariant: !self.strict_guardrails,
                looped_or_timed_out: false,
                final_output: if self.strict_guardrails {
                    "guardrail_blocked_unconfirmed_action".to_string()
                } else {
                    "executed_without_confirmation".to_string()
                },
                steps_executed: 2,
            };
        }
        // - Outros mutantes de tool/contexto: detectados quando guardrails estritos estão ativos
        ReplayExecutionResult {
            detected_or_blocked: self.strict_guardrails,
            violated_invariant: !self.strict_guardrails,
            looped_or_timed_out: false,
            final_output: format!("mcp_replay_{}", mutant.mutation_operator),
            steps_executed: 3,
        }
    }
}
