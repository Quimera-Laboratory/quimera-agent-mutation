use clap::{Parser, Subcommand};
use quimera_agent_mutation::adapters::mcp::McpSandboxAdapter;
use quimera_agent_mutation::complex_system::{analyze_rea_graph, feedback_step_indices};
use quimera_agent_mutation::experiment::{render_experiment_markdown, run_experiment_suite};
use quimera_agent_mutation::exporter::{
    build_rl_feedback, render_html_report, render_markdown_summary, FullCampaignReport,
};
use quimera_agent_mutation::llm_router::BudgetGuard;
use quimera_agent_mutation::oracle::evaluate_outcomes;
use quimera_agent_mutation::quimera_agent_mutator::QuimeraAgentMutator;
use quimera_agent_mutation::rea::{graybox, whitebox};
use quimera_agent_mutation::sandbox_replay::SandboxReplayRunner;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Parser)]
#[command(name = "quimera-agent-mutate")]
#[command(about = "Quimera REA + Complex Systems Mutation Testing for AI Agents")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Engenharia Reversa (REA) de agente a partir de código/prompts ou traces JSONL
    Discover {
        #[arg(long)]
        source: Option<PathBuf>,
        #[arg(long)]
        trace: Option<PathBuf>,
    },
    /// Analisa propriedades de Sistema Complexo sobre um trace JSONL
    Analyze {
        #[arg(long)]
        trace: PathBuf,
    },
    /// Executa campanha completa de mutação de agentes (REA -> CAS -> Replay -> AMS/IRS)
    Mutate {
        #[arg(long)]
        trace: PathBuf,
        #[arg(long, default_value_t = 0.45)]
        entropy_threshold: f32,
        #[arg(long, default_value_t = 64)]
        max_mutants: usize,
        #[arg(long, default_value_t = 0.0)]
        min_ams: f32,
        #[arg(long, default_value_t = true)]
        strict_guardrails: bool,
        #[arg(long, default_value = "markdown")]
        format: String,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        rl_out: Option<PathBuf>,
    },
    /// Reexecuta um único mutante pelo seu hash determinístico
    Replay {
        #[arg(long)]
        trace: PathBuf,
        #[arg(long)]
        mutant_hash: String,
    },
    /// Executa o protocolo experimental comparativo (B0, B1, T1, T2 em A0 vs A1)
    Experiment {
        #[arg(long, default_value_t = 5)]
        reps: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 64)]
        max_mutants: usize,
        #[arg(long, default_value = "markdown")]
        format: String,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Discover { source, trace } => {
            if let Some(src) = source {
                let g = whitebox::extract_from_source(&src).expect("Failed whitebox REA");
                println!("{}", serde_json::to_string_pretty(&g).unwrap());
            } else if let Some(tr) = trace {
                let (g, _, _) = graybox::extract_from_jsonl(&tr).expect("Failed graybox REA");
                println!("{}", serde_json::to_string_pretty(&g).unwrap());
            } else {
                eprintln!("Provide --source or --trace");
                process::exit(2);
            }
        }
        Commands::Analyze { trace } => {
            let (g, _, _) = graybox::extract_from_jsonl(&trace).expect("Failed graybox REA");
            let cas = analyze_rea_graph(&g);
            println!("{}", serde_json::to_string_pretty(&cas).unwrap());
        }
        Commands::Mutate {
            trace,
            entropy_threshold,
            max_mutants,
            min_ams,
            strict_guardrails,
            format,
            out,
            rl_out,
        } => {
            let (g, steps, dag_edges) = graybox::extract_from_jsonl(&trace).expect("Failed trace");
            let cas = analyze_rea_graph(&g);
            let cycles = feedback_step_indices(steps.len(), &dag_edges);
            let mutator = QuimeraAgentMutator::new(entropy_threshold);
            let (mutants, _, _) = mutator.generate_trajectory_mutants(&steps, &dag_edges, &cycles);

            let adapter = McpSandboxAdapter::new("cli-agent", strict_guardrails);
            let budget = BudgetGuard::new(max_mutants, 1_000_000, 10.0);
            let runner = SandboxReplayRunner::new(3, 32);
            let outcomes = runner
                .execute_campaign(&adapter, &mutants, &budget)
                .expect("Baseline failed");
            let metrics = evaluate_outcomes(&outcomes);
            let rl_feedback = build_rl_feedback(&metrics);
            let report = FullCampaignReport {
                system_complexity: cas,
                metrics,
                rl_feedback,
            };

            if let Some(rl_path) = rl_out {
                let rl_json = serde_json::to_string_pretty(&report.rl_feedback).unwrap();
                fs::write(&rl_path, rl_json).expect("Failed to write rl_feedback");
            }

            let rendered = match format.as_str() {
                "json" => serde_json::to_string_pretty(&report).unwrap(),
                "html" => render_html_report(&report),
                _ => render_markdown_summary(&report),
            };
            if let Some(out_path) = out {
                fs::write(&out_path, &rendered).expect("Failed to write report");
            } else {
                println!("{}", rendered);
            }

            if report.metrics.ams < min_ams {
                eprintln!(
                    "CI Gate failed: AMS {:.2}% < min_ams {:.2}%",
                    report.metrics.ams * 100.0,
                    min_ams * 100.0
                );
                process::exit(1);
            }
        }
        Commands::Replay { trace, mutant_hash } => {
            let (_, steps, dag_edges) = graybox::extract_from_jsonl(&trace).expect("Failed trace");
            let cycles = feedback_step_indices(steps.len(), &dag_edges);
            let mutator = QuimeraAgentMutator::new(0.45);
            let (mutants, _, _) = mutator.generate_trajectory_mutants(&steps, &dag_edges, &cycles);
            if let Some(m) = mutants.into_iter().find(|m| m.mutant_hash == mutant_hash) {
                println!("{}", serde_json::to_string_pretty(&m).unwrap());
            } else {
                eprintln!("Mutant hash {} not found", mutant_hash);
                process::exit(1);
            }
        }
        Commands::Experiment {
            reps,
            seed,
            max_mutants,
            format,
        } => {
            let report = run_experiment_suite(reps, seed, max_mutants);
            if format == "json" {
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
            } else {
                println!("{}", render_experiment_markdown(&report));
            }
        }
    }
}
