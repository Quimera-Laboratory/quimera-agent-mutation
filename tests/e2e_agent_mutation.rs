use quimera_agent_mutation::adapters::mcp::McpSandboxAdapter;
use quimera_agent_mutation::adapters::{AgentAdapter, AgentFrameworkKind, ReplayExecutionResult};
use quimera_agent_mutation::complex_system::{analyze_rea_graph, feedback_step_indices};
use quimera_agent_mutation::exporter::{build_rl_feedback, render_markdown_summary, FullCampaignReport};
use quimera_agent_mutation::llm_router::{BudgetGuard, HybridLlmRouter, LlmTier};
use quimera_agent_mutation::oracle::evaluate_outcomes;
use quimera_agent_mutation::quimera_agent_mutator::{AgentMutant, QuimeraAgentMutator};
use quimera_agent_mutation::rea::graybox::parse_jsonl_str;
use quimera_agent_mutation::sandbox_replay::SandboxReplayRunner;

const SAMPLE_TRACE_JSONL: &str = r#"{"step_idx":0,"layer":"PolicyDecision","tool_name":"planner","payload_json":"{\"action\":\"continue\",\"require_confirmation\":true}","parent_step":null}
{"step_idx":1,"layer":"ToolInvocation","tool_name":"sql_query","payload_json":"{\"limit\":25,\"strict\":true}","parent_step":0}
{"step_idx":2,"layer":"MemoryState","tool_name":"rag_memory","payload_json":"{\"verified\":true,\"fact\":\"prod_ro\"}","parent_step":1}
{"step_idx":3,"layer":"PolicyDecision","tool_name":"planner","payload_json":"{\"action\":\"finish\",\"require_confirmation\":true}","parent_step":0}
"#;

struct FailingBaselineAdapter;

impl AgentAdapter for FailingBaselineAdapter {
    fn framework(&self) -> AgentFrameworkKind {
        AgentFrameworkKind::HttpCli
    }
    fn run_baseline(&self) -> ReplayExecutionResult {
        ReplayExecutionResult {
            detected_or_blocked: false,
            violated_invariant: true,
            looped_or_timed_out: false,
            final_output: "baseline_broken".to_string(),
            steps_executed: 1,
        }
    }
    fn run_with_mutant(&self, _mutant: &AgentMutant) -> ReplayExecutionResult {
        self.run_baseline()
    }
}

#[test]
fn e2e_rea_cas_mutation_kills_with_guardrails_and_exposes_survived_without() {
    let (graph, steps, mut dag_edges) = parse_jsonl_str(SAMPLE_TRACE_JSONL).unwrap();
    // Adiciona aresta de feedback loop (3 -> 0)
    dag_edges.push((3, 0));

    let cas = analyze_rea_graph(&graph);
    let cycles = feedback_step_indices(steps.len(), &dag_edges);
    assert!(cycles.contains(&0));
    assert!(cycles.contains(&3));

    let mutator = QuimeraAgentMutator::new(0.40);
    let (mutants, fast, deep) = mutator.generate_trajectory_mutants(&steps, &dag_edges, &cycles);
    assert!(!mutants.is_empty());
    assert!(fast + deep == steps.len());

    // Roteamento híbrido por complexidade
    let router = HybridLlmRouter::new(0.45);
    assert_eq!(router.select_tier(0.10, 0.20), LlmTier::LocalKineCat);
    assert_eq!(router.select_tier(0.80, 0.70), LlmTier::ExternalComplex);

    // Campanha 1: guardrails estritos ativos -> mutantes mortos
    let runner = SandboxReplayRunner::new(3, 32);
    let strict_adapter = McpSandboxAdapter::new("prod-mcp", true);
    let budget_strict = BudgetGuard::new(64, 500_000, 5.0);
    let outcomes_strict = runner
        .execute_campaign(&strict_adapter, &mutants, &budget_strict)
        .unwrap();
    let metrics_strict = evaluate_outcomes(&outcomes_strict);
    assert_eq!(metrics_strict.ams, 1.0);
    assert_eq!(metrics_strict.survived, 0);

    // Campanha 2: sem guardrails -> mutantes sobrevivem / dão timeout
    let weak_adapter = McpSandboxAdapter::new("prod-mcp", false);
    let budget_weak = BudgetGuard::new(64, 500_000, 5.0);
    let outcomes_weak = runner
        .execute_campaign(&weak_adapter, &mutants, &budget_weak)
        .unwrap();
    let metrics_weak = evaluate_outcomes(&outcomes_weak);
    assert!(metrics_weak.survived > 0 || metrics_weak.timeout > 0);
    assert!(metrics_weak.ams < 1.0);

    let rl_batch = build_rl_feedback(&metrics_weak);
    assert!(!rl_batch.survived_vulnerabilities.is_empty());

    let report = FullCampaignReport {
        system_complexity: cas,
        metrics: metrics_weak,
        rl_feedback: rl_batch,
    };
    let md = render_markdown_summary(&report);
    assert!(md.contains("Agent Mutation Score (AMS)"));
}

#[test]
fn e2e_aborts_on_unstable_baseline() {
    let (_, steps, dag_edges) = parse_jsonl_str(SAMPLE_TRACE_JSONL).unwrap();
    let cycles = feedback_step_indices(steps.len(), &dag_edges);
    let mutator = QuimeraAgentMutator::new(0.45);
    let (mutants, _, _) = mutator.generate_trajectory_mutants(&steps, &dag_edges, &cycles);

    let runner = SandboxReplayRunner::new(2, 16);
    let budget = BudgetGuard::new(16, 100_000, 1.0);
    let err = runner
        .execute_campaign(&FailingBaselineAdapter, &mutants, &budget)
        .unwrap_err();
    assert!(err.contains("UNSTABLE_BASELINE"));
}
