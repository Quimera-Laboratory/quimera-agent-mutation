pub mod mcp;
pub mod process_http;

use crate::quimera_agent_mutator::AgentMutant;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentFrameworkKind {
    Mcp,
    LangGraph,
    CrewAi,
    GoogleAdk,
    OpenAiSdk,
    HttpCli,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayExecutionResult {
    pub detected_or_blocked: bool,
    pub violated_invariant: bool,
    pub looped_or_timed_out: bool,
    pub final_output: String,
    pub steps_executed: usize,
}

/// Interface unificada de adaptador para executar um snapshot de agente sob sandbox.
pub trait AgentAdapter: Send + Sync {
    fn framework(&self) -> AgentFrameworkKind;
    fn run_baseline(&self) -> ReplayExecutionResult;
    fn run_with_mutant(&self, mutant: &AgentMutant) -> ReplayExecutionResult;
}
