use crate::rea::{ConfidenceLevel, InferredRule, ReaEdge, ReaEdgeKind, ReaGraph, ReaNode, ReaNodeKind};
use std::fs;
use std::path::Path;

/// Extrator White-box: analisa código-fonte, prompts de sistema e schemas MCP/JSON.
pub fn extract_from_source(path: &Path) -> std::io::Result<ReaGraph> {
    let mut graph = ReaGraph {
        mode: "whitebox".to_string(),
        ..Default::default()
    };

    graph.add_node_unique(ReaNode {
        id: "agent:root".to_string(),
        kind: ReaNodeKind::Agent,
        label: "PrimaryAgent".to_string(),
        confidence: ConfidenceLevel::High,
    });

    if path.is_file() {
        let content = fs::read_to_string(path)?;
        scan_text_into_graph(&content, &mut graph);
    } else if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let p = entry.path();
            if p.is_file() {
                if let Ok(content) = fs::read_to_string(&p) {
                    scan_text_into_graph(&content, &mut graph);
                }
            }
        }
    }

    Ok(graph)
}

pub fn scan_text_into_graph(content: &str, graph: &mut ReaGraph) {
    // Detecta ferramentas declaradas em JSON/MCP/LangGraph/CrewAI
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(name) = extract_quoted_after(trimmed, "\"tool_name\":")
            .or_else(|| extract_quoted_after(trimmed, "\"name\":"))
        {
            let tid = format!("tool:{}", name);
            graph.add_node_unique(ReaNode {
                id: tid.clone(),
                kind: ReaNodeKind::Tool,
                label: name.to_string(),
                confidence: ConfidenceLevel::High,
            });
            graph.add_edge_unique(ReaEdge {
                from: "agent:root".to_string(),
                to: tid,
                kind: ReaEdgeKind::Calls,
            });
        }

        if trimmed.contains("require_confirmation") || trimmed.contains("NEVER") || trimmed.contains("must not") {
            let rule_id = format!("rule:{}", graph.inferred_rules.len() + 1);
            graph.add_node_unique(ReaNode {
                id: rule_id.clone(),
                kind: ReaNodeKind::Policy,
                label: trimmed.chars().take(64).collect(),
                confidence: ConfidenceLevel::High,
            });
            graph.add_edge_unique(ReaEdge {
                from: rule_id.clone(),
                to: "agent:root".to_string(),
                kind: ReaEdgeKind::Enforces,
            });
            graph.inferred_rules.push(InferredRule {
                id: rule_id,
                description: trimmed.to_string(),
                target_node: "agent:root".to_string(),
                confidence: ConfidenceLevel::High,
            });
        }

        if trimmed.contains("rag") || trimmed.contains("memory") || trimmed.contains("vector_store") {
            graph.add_node_unique(ReaNode {
                id: "memory:rag".to_string(),
                kind: ReaNodeKind::Memory,
                label: "RAG/Memory".to_string(),
                confidence: ConfidenceLevel::Medium,
            });
            graph.add_edge_unique(ReaEdge {
                from: "agent:root".to_string(),
                to: "memory:rag".to_string(),
                kind: ReaEdgeKind::Reads,
            });
        }
    }
}

fn extract_quoted_after<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let pos = line.find(key)?;
    let rest = line[pos + key.len()..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    let val = &rest[..end];
    if val.is_empty() {
        None
    } else {
        Some(val)
    }
}
