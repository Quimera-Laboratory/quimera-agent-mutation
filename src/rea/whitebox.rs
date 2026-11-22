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
    let mut expect_python_tool_fn = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // 1. JSON / MCP / OpenAI SDK tool declarations
        if let Some(name) = extract_quoted_after(trimmed, "\"tool_name\":")
            .or_else(|| extract_quoted_after(trimmed, "\"name\":"))
            .or_else(|| extract_quoted_after(trimmed, "server.tool("))
        {
            register_tool(graph, name, ConfidenceLevel::High);
        }

        // 2. Python decorators (@tool) / CrewAI / LangGraph / ADK
        if trimmed.starts_with("@tool") || trimmed.contains("StructuredTool") {
            expect_python_tool_fn = true;
        } else if expect_python_tool_fn && trimmed.starts_with("def ") {
            if let Some(fn_name) = trimmed
                .strip_prefix("def ")
                .and_then(|s| s.split('(').next())
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                register_tool(graph, fn_name, ConfidenceLevel::High);
            }
            expect_python_tool_fn = false;
        }

        // 3. Multi-agent / LangGraph / CrewAI edges & subagents
        if trimmed.contains("Agent(") || trimmed.contains("StateGraph(") {
            graph.add_node_unique(ReaNode {
                id: "agent:subagent".to_string(),
                kind: ReaNodeKind::Agent,
                label: "SubAgent/Graph".to_string(),
                confidence: ConfidenceLevel::Medium,
            });
            graph.add_edge_unique(ReaEdge {
                from: "agent:root".to_string(),
                to: "agent:subagent".to_string(),
                kind: ReaEdgeKind::Delegates,
            });
        }
        if trimmed.contains(".add_conditional_edges(") || trimmed.contains(".add_edge(") {
            graph.add_edge_unique(ReaEdge {
                from: "agent:subagent".to_string(),
                to: "agent:root".to_string(),
                kind: ReaEdgeKind::Delegates,
            });
        }

        // 4. Policy guardrails & system prompt invariants
        if trimmed.contains("require_confirmation")
            || trimmed.contains("NEVER")
            || trimmed.contains("must not")
            || trimmed.contains("DO NOT")
            || trimmed.contains("forbidden")
        {
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

        // 5. Memory / RAG stores
        if trimmed.contains("rag")
            || trimmed.contains("memory")
            || trimmed.contains("vector_store")
            || trimmed.contains("retriever")
        {
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

fn register_tool(graph: &mut ReaGraph, name: &str, confidence: ConfidenceLevel) {
    let tid = format!("tool:{}", name);
    graph.add_node_unique(ReaNode {
        id: tid.clone(),
        kind: ReaNodeKind::Tool,
        label: name.to_string(),
        confidence,
    });
    graph.add_edge_unique(ReaEdge {
        from: "agent:root".to_string(),
        to: tid,
        kind: ReaEdgeKind::Calls,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_langgraph_python_mcp_and_guardrails() {
        let sample = r#"
@tool
def query_ledger(limit: int):
    pass
graph = StateGraph(AgentState)
graph.add_conditional_edges("a", route)
# NEVER execute transfer without require_confirmation
retriever = vector_store.as_retriever()
"#;
        let mut g = ReaGraph::default();
        g.add_node_unique(ReaNode {
            id: "agent:root".into(),
            kind: ReaNodeKind::Agent,
            label: "Root".into(),
            confidence: ConfidenceLevel::High,
        });
        scan_text_into_graph(sample, &mut g);
        assert!(g.nodes.iter().any(|n| n.id == "tool:query_ledger"));
        assert!(g.nodes.iter().any(|n| n.id == "memory:rag"));
        assert!(g.nodes.iter().any(|n| n.id == "agent:subagent"));
        assert!(!g.inferred_rules.is_empty());
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
