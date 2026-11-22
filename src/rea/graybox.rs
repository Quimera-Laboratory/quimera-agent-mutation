use crate::quimera_agent_mutator::{AgentGermLayer, AgentStepTrace};
use crate::rea::{ConfidenceLevel, ReaEdge, ReaEdgeKind, ReaGraph, ReaNode, ReaNodeKind};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Deserialize)]
struct RawTraceRecord {
    step_idx: usize,
    layer: String,
    tool_name: String,
    payload_json: String,
    #[serde(default)]
    parent_step: Option<usize>,
}

/// Extrator Gray-box: converte traces JSONL/OpenTelemetry em ReaGraph + AgentStepTrace + DAG edges.
pub fn extract_from_jsonl(
    path: &Path,
) -> Result<(ReaGraph, Vec<AgentStepTrace>, Vec<(usize, usize)>), String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    parse_jsonl_str(&content)
}

pub fn parse_jsonl_str(
    content: &str,
) -> Result<(ReaGraph, Vec<AgentStepTrace>, Vec<(usize, usize)>), String> {
    let mut graph = ReaGraph {
        mode: "graybox".to_string(),
        ..Default::default()
    };
    graph.add_node_unique(ReaNode {
        id: "agent:root".to_string(),
        kind: ReaNodeKind::Agent,
        label: "TracedAgent".to_string(),
        confidence: ConfidenceLevel::High,
    });

    let mut traces = Vec::new();
    let mut dag_edges = Vec::new();

    for (line_no, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let rec: RawTraceRecord = serde_json::from_str(trimmed)
            .map_err(|e| format!("Invalid trace JSON at line {}: {}", line_no + 1, e))?;

        let layer = match rec.layer.as_str() {
            "PolicyDecision" | "policy" => AgentGermLayer::PolicyDecision,
            "MemoryState" | "memory" => AgentGermLayer::MemoryState,
            _ => AgentGermLayer::ToolInvocation,
        };

        let node_id = format!("tool:{}", rec.tool_name);
        let kind = match layer {
            AgentGermLayer::PolicyDecision => ReaNodeKind::Policy,
            AgentGermLayer::MemoryState => ReaNodeKind::Memory,
            AgentGermLayer::ToolInvocation => ReaNodeKind::Tool,
        };

        graph.add_node_unique(ReaNode {
            id: node_id.clone(),
            kind,
            label: rec.tool_name.clone(),
            confidence: ConfidenceLevel::High,
        });
        graph.add_edge_unique(ReaEdge {
            from: "agent:root".to_string(),
            to: node_id,
            kind: ReaEdgeKind::Calls,
        });

        if let Some(parent) = rec.parent_step {
            dag_edges.push((parent, rec.step_idx));
        } else if rec.step_idx > 0 {
            dag_edges.push((rec.step_idx - 1, rec.step_idx));
        }

        // Embedding determinístico leve de 128 floats a partir dos bytes do payload
        let mut latent = vec![0.0_f32; 128];
        for (i, b) in rec.payload_json.bytes().enumerate() {
            latent[i % 128] += (b as f32) / 255.0;
        }

        traces.push(AgentStepTrace {
            step_idx: rec.step_idx,
            layer,
            tool_name: rec.tool_name,
            payload_json: rec.payload_json,
            latent_state_128: latent,
        });
    }

    Ok((graph, traces, dag_edges))
}
