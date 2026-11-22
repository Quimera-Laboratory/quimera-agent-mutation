use crate::rea::{ConfidenceLevel, InferredRule, ReaEdge, ReaEdgeKind, ReaGraph, ReaNode, ReaNodeKind};

/// Resultado de uma sonda ativa (Black-box probing).
#[derive(Clone, Debug)]
pub struct ProbeObservation {
    pub probe_name: String,
    pub tool_triggered: Option<String>,
    pub blocked_by_policy: bool,
    pub response_snippet: String,
}

/// Extrator Black-box: infere fronteiras e invariantes via sondagem ativa controlada.
pub fn infer_from_probes(agent_label: &str, observations: &[ProbeObservation]) -> ReaGraph {
    let mut graph = ReaGraph {
        mode: "blackbox".to_string(),
        ..Default::default()
    };
    let root_id = format!("agent:{}", agent_label);
    graph.add_node_unique(ReaNode {
        id: root_id.clone(),
        kind: ReaNodeKind::Agent,
        label: agent_label.to_string(),
        confidence: ConfidenceLevel::Medium,
    });

    for obs in observations {
        if let Some(tool) = &obs.tool_triggered {
            let tid = format!("tool:{}", tool);
            graph.add_node_unique(ReaNode {
                id: tid.clone(),
                kind: ReaNodeKind::Tool,
                label: tool.clone(),
                confidence: ConfidenceLevel::Medium,
            });
            graph.add_edge_unique(ReaEdge {
                from: root_id.clone(),
                to: tid,
                kind: ReaEdgeKind::Calls,
            });
        }
        if obs.blocked_by_policy {
            let pid = format!("policy:{}", obs.probe_name);
            graph.add_node_unique(ReaNode {
                id: pid.clone(),
                kind: ReaNodeKind::Policy,
                label: obs.probe_name.clone(),
                confidence: ConfidenceLevel::Medium,
            });
            graph.add_edge_unique(ReaEdge {
                from: pid.clone(),
                to: root_id.clone(),
                kind: ReaEdgeKind::Enforces,
            });
            graph.inferred_rules.push(InferredRule {
                id: pid,
                description: format!("Policy boundary triggered on probe {}", obs.probe_name),
                target_node: root_id.clone(),
                confidence: ConfidenceLevel::Medium,
            });
        }
    }

    graph
}
