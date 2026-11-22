pub mod blackbox;
pub mod graybox;
pub mod whitebox;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReaNodeKind {
    Agent,
    Tool,
    Memory,
    Policy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReaEdgeKind {
    Calls,
    Reads,
    Writes,
    Delegates,
    Enforces,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    High,
    Medium,
    Low,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReaNode {
    pub id: String,
    pub kind: ReaNodeKind,
    pub label: String,
    pub confidence: ConfidenceLevel,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReaEdge {
    pub from: String,
    pub to: String,
    pub kind: ReaEdgeKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferredRule {
    pub id: String,
    pub description: String,
    pub target_node: String,
    pub confidence: ConfidenceLevel,
}

/// Grafo causal unificado extraído via REA (Reverse Engineer Anything).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReaGraph {
    pub mode: String,
    pub nodes: Vec<ReaNode>,
    pub edges: Vec<ReaEdge>,
    pub inferred_rules: Vec<InferredRule>,
}

impl ReaGraph {
    pub fn add_node_unique(&mut self, node: ReaNode) {
        if !self.nodes.iter().any(|n| n.id == node.id) {
            self.nodes.push(node);
        }
    }

    pub fn add_edge_unique(&mut self, edge: ReaEdge) {
        if !self
            .edges
            .iter()
            .any(|e| e.from == edge.from && e.to == edge.to && e.kind == edge.kind)
        {
            self.edges.push(edge);
        }
    }

    /// Converte as arestas do grafo REA para índices numéricos `(src_idx, dst_idx)`.
    pub fn to_indexed_edges(&self) -> (HashMap<String, usize>, Vec<(usize, usize)>) {
        let mut idx_map = HashMap::new();
        for (idx, n) in self.nodes.iter().enumerate() {
            idx_map.insert(n.id.clone(), idx);
        }
        let mut edges = Vec::new();
        for e in &self.edges {
            if let (Some(&u), Some(&v)) = (idx_map.get(&e.from), idx_map.get(&e.to)) {
                edges.push((u, v));
            }
        }
        (idx_map, edges)
    }
}
