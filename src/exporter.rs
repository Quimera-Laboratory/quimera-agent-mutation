use crate::complex_system::ComplexSystemReport;
use crate::oracle::{CampaignMetrics, MutantVerdict};
use serde::{Deserialize, Serialize};

/// Lote de retroalimentação para EmaFusedMutationRayleighLagrangian (KineCat-1B-100MB).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RlFeedbackBatch {
    pub killed_states: Vec<String>,
    pub equiv_states: Vec<String>,
    pub survived_vulnerabilities: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FullCampaignReport {
    pub system_complexity: ComplexSystemReport,
    pub metrics: CampaignMetrics,
    pub rl_feedback: RlFeedbackBatch,
}

pub fn build_rl_feedback(metrics: &CampaignMetrics) -> RlFeedbackBatch {
    let mut killed_states = Vec::new();
    let mut equiv_states = Vec::new();
    let mut survived_vulnerabilities = Vec::new();

    for ev in &metrics.evaluations {
        let tag = format!("{}:{}:step_{}", ev.mutant_hash, ev.mutation_operator, ev.target_step);
        match ev.verdict {
            MutantVerdict::Killed => killed_states.push(tag),
            MutantVerdict::Equivalent => equiv_states.push(tag),
            MutantVerdict::Survived | MutantVerdict::Timeout => survived_vulnerabilities.push(tag),
        }
    }

    RlFeedbackBatch {
        killed_states,
        equiv_states,
        survived_vulnerabilities,
    }
}

pub fn render_markdown_summary(report: &FullCampaignReport) -> String {
    format!(
        "# Quimera Agent Mutation Report\n\n\
         - **System Complexity Score**: {:.3}\n\
         - **Feedback Loops**: {}\n\
         - **Total Mutants**: {} | **Killed**: {} | **Survived**: {} | **Timeout**: {} | **Equivalent**: {}\n\
         - **Agent Mutation Score (AMS)**: {:.2}%\n\
         - **Índice de Resiliência Sistêmica (IRS)**: {:.2}%\n",
        report.system_complexity.system_complexity_score,
        report.system_complexity.feedback_loops.len(),
        report.metrics.total_mutants,
        report.metrics.killed,
        report.metrics.survived,
        report.metrics.timeout,
        report.metrics.equivalent,
        report.metrics.ams * 100.0,
        report.metrics.irs * 100.0,
    )
}

pub fn render_html_report(report: &FullCampaignReport) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
        <title>Quimera Agent Mutation Report</title>\
        <style>body{{background:#050505;color:#fff;font-family:monospace;padding:24px}}\
        h1{{color:#CCFF00}} .card{{border:1px solid #222;padding:16px;margin:12px 0;background:#0a0a0a}}</style>\
        </head><body><h1>QUIMERA AGENT MUTATION // REPORT</h1>\
        <div class=\"card\">System Complexity: {:.3} | Feedback Loops: {}</div>\
        <div class=\"card\">Mutants: {} | Killed: {} | Survived: {} | Timeout: {} | Equivalent: {}</div>\
        <div class=\"card\"><strong>AMS: {:.2}%</strong> | <strong>IRS: {:.2}%</strong></div>\
        </body></html>",
        report.system_complexity.system_complexity_score,
        report.system_complexity.feedback_loops.len(),
        report.metrics.total_mutants,
        report.metrics.killed,
        report.metrics.survived,
        report.metrics.timeout,
        report.metrics.equivalent,
        report.metrics.ams * 100.0,
        report.metrics.irs * 100.0,
    )
}
