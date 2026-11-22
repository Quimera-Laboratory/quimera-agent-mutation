use crate::rea::ReaGraph;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComplexSystemReport {
    pub node_count: usize,
    pub edge_count: usize,
    pub feedback_loops: Vec<Vec<String>>,
    pub centrality_ranking: Vec<(String, f32)>,
    /// Índice normalizado [0.0, 1.0] usado pelo roteador híbrido de LLM.
    pub system_complexity_score: f32,
}

/// Analisa o grafo REA sob a ótica de Sistemas Complexos Adaptativos (CAS).
pub fn analyze_rea_graph(graph: &ReaGraph) -> ComplexSystemReport {
    let n = graph.nodes.len();
    let (_, edges) = graph.to_indexed_edges();
    let mut adj = vec![Vec::new(); n];
    for &(u, v) in &edges {
        adj[u].push(v);
    }

    let sccs = tarjan_scc(n, &adj);
    let mut feedback_loops = Vec::new();
    for scc in sccs {
        let is_cycle = scc.len() > 1 || (scc.len() == 1 && adj[scc[0]].contains(&scc[0]));
        if is_cycle {
            let labels = scc.into_iter().map(|i| graph.nodes[i].id.clone()).collect();
            feedback_loops.push(labels);
        }
    }

    let centrality = brandes_betweenness(n, &adj);
    let mut centrality_ranking: Vec<(String, f32)> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (node.id.clone(), centrality[i]))
        .collect();
    centrality_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Score de complexidade sistêmica combina densidade de arestas, ciclos e regras
    let density = if n > 1 {
        (edges.len() as f32) / ((n * (n - 1)) as f32)
    } else {
        0.0
    };
    let loop_factor = (feedback_loops.len() as f32 * 0.25).min(0.5);
    let scale_factor = (n as f32 / 20.0).min(0.3);
    let system_complexity_score = (density * 0.4 + loop_factor + scale_factor).clamp(0.0, 1.0);

    ComplexSystemReport {
        node_count: n,
        edge_count: edges.len(),
        feedback_loops,
        centrality_ranking,
        system_complexity_score,
    }
}

/// Identifica índices de passos pertencentes a ciclos de retroalimentação no DAG de execução.
pub fn feedback_step_indices(num_steps: usize, dag_edges: &[(usize, usize)]) -> HashSet<usize> {
    let mut adj = vec![Vec::new(); num_steps];
    for &(u, v) in dag_edges {
        if u < num_steps && v < num_steps {
            adj[u].push(v);
        }
    }
    let mut in_cycle = HashSet::new();
    for scc in tarjan_scc(num_steps, &adj) {
        if scc.len() > 1 || (scc.len() == 1 && adj[scc[0]].contains(&scc[0])) {
            for idx in scc {
                in_cycle.insert(idx);
            }
        }
    }
    in_cycle
}

fn tarjan_scc(n: usize, adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    struct State {
        index: usize,
        indices: Vec<Option<usize>>,
        lowlink: Vec<usize>,
        on_stack: Vec<bool>,
        stack: Vec<usize>,
        sccs: Vec<Vec<usize>>,
    }

    fn strongconnect(v: usize, adj: &[Vec<usize>], st: &mut State) {
        st.indices[v] = Some(st.index);
        st.lowlink[v] = st.index;
        st.index += 1;
        st.stack.push(v);
        st.on_stack[v] = true;

        for &w in &adj[v] {
            if st.indices[w].is_none() {
                strongconnect(w, adj, st);
                st.lowlink[v] = st.lowlink[v].min(st.lowlink[w]);
            } else if st.on_stack[w] {
                st.lowlink[v] = st.lowlink[v].min(st.indices[w].unwrap());
            }
        }

        if st.lowlink[v] == st.indices[v].unwrap() {
            let mut scc = Vec::new();
            while let Some(w) = st.stack.pop() {
                st.on_stack[w] = false;
                scc.push(w);
                if w == v {
                    break;
                }
            }
            st.sccs.push(scc);
        }
    }

    let mut st = State {
        index: 0,
        indices: vec![None; n],
        lowlink: vec![0; n],
        on_stack: vec![false; n],
        stack: Vec::new(),
        sccs: Vec::new(),
    };
    for v in 0..n {
        if st.indices[v].is_none() {
            strongconnect(v, adj, &mut st);
        }
    }
    st.sccs
}

fn brandes_betweenness(n: usize, adj: &[Vec<usize>]) -> Vec<f32> {
    let mut cb = vec![0.0_f32; n];
    for s in 0..n {
        let mut stack = Vec::new();
        let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma = vec![0.0_f32; n];
        sigma[s] = 1.0;
        let mut dist = vec![-1_i32; n];
        dist[s] = 0;
        let mut q = VecDeque::new();
        q.push_back(s);

        while let Some(v) = q.pop_front() {
            stack.push(v);
            for &w in &adj[v] {
                if dist[w] < 0 {
                    q.push_back(w);
                    dist[w] = dist[v] + 1;
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    pred[w].push(v);
                }
            }
        }

        let mut delta = vec![0.0_f32; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                if sigma[w] > 0.0 {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
            }
            if w != s {
                cb[w] += delta[w];
            }
        }
    }
    let norm = if n > 2 { ((n - 1) * (n - 2)) as f32 } else { 1.0 };
    cb.into_iter().map(|v| v / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rea::{ConfidenceLevel, ReaEdge, ReaEdgeKind, ReaNode, ReaNodeKind};

    #[test]
    fn detects_feedback_loops_and_computes_complexity() {
        let mut g = ReaGraph::default();
        for id in ["a", "b", "c"] {
            g.add_node_unique(ReaNode {
                id: id.to_string(),
                kind: ReaNodeKind::Agent,
                label: id.to_string(),
                confidence: ConfidenceLevel::High,
            });
        }
        g.add_edge_unique(ReaEdge { from: "a".into(), to: "b".into(), kind: ReaEdgeKind::Delegates });
        g.add_edge_unique(ReaEdge { from: "b".into(), to: "a".into(), kind: ReaEdgeKind::Delegates });
        g.add_edge_unique(ReaEdge { from: "b".into(), to: "c".into(), kind: ReaEdgeKind::Calls });

        let report = analyze_rea_graph(&g);
        assert_eq!(report.feedback_loops.len(), 1);
        assert!(report.system_complexity_score > 0.25);
    }
}
