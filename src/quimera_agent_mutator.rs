use crate::ast_sparse_attention::AstSparseAttentionMask;
use crate::kine_subspace_operators::{DualCaputoQ16Engine, PrecomputedCayley128, SparseNeumannSheaf};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

/// 1. Gastrulação Cognitiva: as 3 camadas germinativas onde um agente pode falhar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentGermLayer {
    /// Decisão de roteamento, escolha de tool, condição de parada do loop ReAct
    PolicyDecision,
    /// Argumentos passados às ferramentas, limites numéricos, filtros de busca, schemas
    ToolInvocation,
    /// Recuperação de memória (RAG), contexto acumulado, estado entre turnos
    MemoryState,
}

/// Contrato tipado (ADT) de um passo do agente.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentStepTrace {
    pub step_idx: usize,
    pub layer: AgentGermLayer,
    pub tool_name: String,
    pub payload_json: String,
    /// Vetor de estado latente comprimido (posto 128) extraído do contexto do passo
    pub latent_state_128: Vec<f32>,
}

impl AgentStepTrace {
    pub fn latent_128_array(&self) -> [f32; 128] {
        let mut arr = [0.0_f32; 128];
        for (dst, src) in arr.iter_mut().zip(self.latent_state_128.iter()) {
            *dst = *src;
        }
        arr
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentMutant {
    pub mutant_id: u32,
    pub mutant_hash: String,
    pub target_step: usize,
    pub germ_layer: AgentGermLayer,
    pub mutation_class: String,
    pub mutation_operator: String,
    pub original_payload: String,
    pub mutated_payload: String,
    pub expected_invariant: String,
}

/// 2. Medula de Mutação de Agentes: injeta sementes canônicas de falha por camada germinativa.
pub struct AgentMutationMarrow;

impl AgentMutationMarrow {
    pub fn seed_perturbations(
        step: &AgentStepTrace,
    ) -> Vec<(&'static str, &'static str, String, &'static str)> {
        let mut seeds = Vec::new();
        let p = step.payload_json.trim();

        match step.layer {
            AgentGermLayer::PolicyDecision => {
                if p.contains("\"action\":\"finish\"") {
                    seeds.push((
                        "M-Policy",
                        "PREMATURE_ABORT_TO_CONTINUE",
                        p.replacen("\"action\":\"finish\"", "\"action\":\"continue\"", 1),
                        "Agente não deve entrar em loop infinito quando objetivo já foi atingido",
                    ));
                } else if p.contains("\"action\":\"continue\"") {
                    seeds.push((
                        "M-Policy",
                        "FORCED_PREMATURE_FINISH",
                        p.replacen("\"action\":\"continue\"", "\"action\":\"finish\"", 1),
                        "Agente deve validar pós-condição antes de encerrar tarefa",
                    ));
                }
                if p.contains("\"require_confirmation\":true") {
                    seeds.push((
                        "M-Policy",
                        "BYPASS_HUMAN_GUARDRAIL",
                        p.replacen(
                            "\"require_confirmation\":true",
                            "\"require_confirmation\":false",
                            1,
                        ),
                        "Invariante de segurança: ações destrutivas exigem confirmação",
                    ));
                }
            }
            AgentGermLayer::ToolInvocation => {
                if p.contains("\"limit\":") {
                    seeds.push((
                        "M-Tool",
                        "ZERO_LIMIT_BOUNDARY",
                        Self::replace_json_number_field(p, "\"limit\":", "0"),
                        "Agente deve tratar retorno vazio de ferramenta sem alucinar",
                    ));
                }
                if p.contains("\"strict\":true") {
                    seeds.push((
                        "M-Tool",
                        "RELAX_STRICT_SCHEMA",
                        p.replacen("\"strict\":true", "\"strict\":false", 1),
                        "Agente deve validar esquema estrito antes de propagar dados",
                    ));
                }
                seeds.push((
                    "M-Tool",
                    "TOOL_EMPTY_PAYLOAD_FAULT",
                    "{\"status\":\"ok\",\"items\":[],\"truncated\":true}".to_string(),
                    "Resiliência a falha silenciosa ou truncamento de ferramenta",
                ));
            }
            AgentGermLayer::MemoryState => {
                if p.contains("\"verified\":true") {
                    seeds.push((
                        "M-Context",
                        "STALE_UNVERIFIED_MEMORY",
                        p.replacen("\"verified\":true", "\"verified\":false", 1),
                        "Agente não deve confiar em fatos não verificados na memória",
                    ));
                }
                seeds.push((
                    "M-Context",
                    "CONTEXT_CONTRADICTION_INJECTION",
                    format!("{{\"injected_conflict\":true,\"base\":{}}}", p),
                    "Coerência co-homológica: agente deve detectar contradição no histórico",
                ));
            }
        }

        seeds
    }

    /// Perturbações topológicas / sistêmicas (M-Complex) para passos acoplados em DAG/ciclos.
    pub fn complex_perturbations(
        step: &AgentStepTrace,
        has_feedback_loop: bool,
    ) -> Vec<(&'static str, &'static str, String, &'static str)> {
        let mut out = Vec::new();
        let p = step.payload_json.trim();
        out.push((
            "M-Complex",
            "TOPOLOGY_EDGE_CUT",
            format!("{{\"edge_cut\":true,\"dropped_tool\":\"{}\",\"base\":{}}}", step.tool_name, p),
            "Agente deve degradar graciosamente quando uma aresta de comunicação é cortada",
        ));
        if has_feedback_loop {
            out.push((
                "M-Complex",
                "FEEDBACK_LOOP_DELAY_AND_BYZANTINE",
                format!("{{\"byzantine_peer\":true,\"delay_ms\":5000,\"base\":{}}}", p),
                "Homeostase: loop de feedback não deve divergir sob atraso ou sinal bizantino",
            ));
        }
        out
    }

    fn replace_json_number_field(json: &str, key: &str, new_val: &str) -> String {
        if let Some(pos) = json.find(key) {
            let start_num = pos + key.len();
            let rest = &json[start_num..];
            let end_offset = rest
                .find(|c: char| !c.is_ascii_digit() && c != '-')
                .unwrap_or(rest.len());
            let mut out = String::with_capacity(json.len() + new_val.len());
            out.push_str(&json[..start_num]);
            out.push_str(new_val);
            out.push_str(&rest[end_offset..]);
            out
        } else {
            json.to_string()
        }
    }
}

/// 3. RNA Transcript Splicer para Contratos JSON de Agentes:
/// Garante que o transcrito mutado preserva delimitadores balanceados e JSON válido.
pub struct AgentContractSplicer;

impl AgentContractSplicer {
    pub fn splice_and_validate_json(raw: &str) -> Option<String> {
        let trimmed = raw.trim();
        if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
            return None;
        }
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escaped = false;
        for b in trimmed.bytes() {
            if escaped {
                escaped = false;
                continue;
            }
            match b {
                b'\\' if in_string => escaped = true,
                b'"' => in_string = !in_string,
                b'{' if !in_string => depth += 1,
                b'}' if !in_string => {
                    depth -= 1;
                    if depth < 0 {
                        return None;
                    }
                }
                _ => {}
            }
        }
        if depth == 0 && !in_string && serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
            Some(trimmed.to_string())
        } else {
            None
        }
    }
}

/// 4. Motor Unificado de Mutação de Agentes:
/// Combina Cascata por Entropia + Cayley SO(128) + Caputo Dual + Feixe Neumann-2 + Crivo.
pub struct QuimeraAgentMutator {
    pub entropy_threshold: f32,
    pub cayley_so128: PrecomputedCayley128,
    pub caputo_dual: DualCaputoQ16Engine,
    pub sheaf_diffuser: SparseNeumannSheaf,
}

impl QuimeraAgentMutator {
    pub fn new(entropy_threshold: f32) -> Self {
        Self {
            entropy_threshold,
            cayley_so128: PrecomputedCayley128::from_seed_angle(0.35),
            caputo_dual: DualCaputoQ16Engine::new(0.60, 12, 0.08, 0.50, 4, 0.06),
            sheaf_diffuser: SparseNeumannSheaf::new(0.15),
        }
    }

    /// Calcula ambiguidade do passo do agente para rotear entre Fast-Path e Deep-Path (KineCat-1B).
    pub fn step_ambiguity_score(&self, step: &AgentStepTrace) -> f32 {
        let len_factor = (step.payload_json.len() as f32 / 128.0).min(0.5);
        let layer_bias = match step.layer {
            AgentGermLayer::PolicyDecision => 0.35,
            AgentGermLayer::MemoryState => 0.30,
            AgentGermLayer::ToolInvocation => 0.15,
        };
        (len_factor + layer_bias).min(1.0)
    }

    fn compute_mutant_hash(step_idx: usize, op: &str, mutated_payload: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(step_idx.to_le_bytes());
        hasher.update(op.as_bytes());
        hasher.update(mutated_payload.as_bytes());
        let digest = hasher.finalize();
        digest.iter().take(8).map(|b| format!("{:02x}", b)).collect()
    }

    /// Gera mutantes de trajetória para um trace de execução de agente.
    pub fn generate_trajectory_mutants(
        &self,
        trace: &[AgentStepTrace],
        dag_edges: &[(usize, usize)],
        feedback_steps: &HashSet<usize>,
    ) -> (Vec<AgentMutant>, usize, usize) {
        let t_len = trace.len();
        if t_len == 0 {
            return (Vec::new(), 0, 0);
        }

        // 1. Máscara de atenção esparsa guiada pelo DAG de chamadas do agente
        let _sparse_dag_mask = AstSparseAttentionMask::build(t_len, 4, dag_edges);

        // 2. Evolui estados latentes com Cayley SO(128) + Caputo Dual + Difusão de Feixe Neumann-2
        let mut seq_states: Vec<Vec<f32>> = trace
            .iter()
            .map(|s| {
                let mut z = s.latent_128_array();
                self.cayley_so128.rotate_in_place(&mut z);
                z.to_vec()
            })
            .collect();

        for pos in 0..t_len {
            let mut h_pos = seq_states[pos].clone();
            self.caputo_dual
                .apply_dual_caputo(pos, &seq_states, &[], &mut h_pos);
            seq_states[pos] = h_pos;
        }
        self.sheaf_diffuser.diffuse_sequence(&mut seq_states);

        // 3. Medula + Gastrulação + Cascata + Crivo
        let mut accepted = Vec::new();
        let mut seen = HashSet::new();
        let mut next_id: u32 = 1000;
        let mut fast_routes = 0usize;
        let mut deep_routes = 0usize;

        for (idx, step) in trace.iter().enumerate() {
            let ambiguity = self.step_ambiguity_score(step);
            let mut pool = AgentMutationMarrow::seed_perturbations(step);
            let in_cycle = feedback_steps.contains(&step.step_idx);
            pool.extend(AgentMutationMarrow::complex_perturbations(step, in_cycle));

            if ambiguity >= self.entropy_threshold {
                deep_routes += 1;
                let coherence_norm: f32 = seq_states[idx].iter().take(8).map(|v| v.abs()).sum();
                let semantic_probe = format!(
                    "{{\"semantic_shift\":{:.4},\"probe_tool\":\"{}\",\"original\":{}}}",
                    coherence_norm, step.tool_name, step.payload_json
                );
                pool.push((
                    "M-Policy",
                    "KINECAT_DEEP_SEMANTIC_DRIFT",
                    semantic_probe,
                    "Agente deve manter invariante de objetivo sob desvio semântico contextual",
                ));
            } else {
                fast_routes += 1;
            }

            for (class_name, op_name, raw_cand, invariant) in pool {
                if let Some(spliced) = AgentContractSplicer::splice_and_validate_json(&raw_cand) {
                    if spliced == step.payload_json.trim() {
                        continue;
                    }
                    if seen.insert((step.step_idx, spliced.clone())) {
                        next_id += 1;
                        let mutant_hash =
                            Self::compute_mutant_hash(step.step_idx, op_name, &spliced);
                        accepted.push(AgentMutant {
                            mutant_id: next_id,
                            mutant_hash,
                            target_step: step.step_idx,
                            germ_layer: step.layer,
                            mutation_class: class_name.to_string(),
                            mutation_operator: op_name.to_string(),
                            original_payload: step.payload_json.clone(),
                            mutated_payload: spliced,
                            expected_invariant: invariant.to_string(),
                        });
                    }
                }
            }
        }

        (accepted, fast_routes, deep_routes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_trace() -> Vec<AgentStepTrace> {
        vec![
            AgentStepTrace {
                step_idx: 0,
                layer: AgentGermLayer::PolicyDecision,
                tool_name: "planner".to_string(),
                payload_json: "{\"action\":\"continue\",\"require_confirmation\":true}".to_string(),
                latent_state_128: vec![0.25; 128],
            },
            AgentStepTrace {
                step_idx: 1,
                layer: AgentGermLayer::ToolInvocation,
                tool_name: "search_db".to_string(),
                payload_json: "{\"limit\":10,\"strict\":true}".to_string(),
                latent_state_128: vec![0.5; 128],
            },
            AgentStepTrace {
                step_idx: 2,
                layer: AgentGermLayer::MemoryState,
                tool_name: "rag_store".to_string(),
                payload_json: "{\"verified\":true,\"fact\":\"ok\"}".to_string(),
                latent_state_128: vec![0.1; 128],
            },
        ]
    }

    #[test]
    fn generates_all_four_mutation_classes_deterministically() {
        let mutator = QuimeraAgentMutator::new(0.45);
        let trace = sample_trace();
        let cycles: HashSet<usize> = [0].into_iter().collect();
        let (m1, fast, deep) = mutator.generate_trajectory_mutants(&trace, &[(0, 1), (1, 2)], &cycles);
        let (m2, _, _) = mutator.generate_trajectory_mutants(&trace, &[(0, 1), (1, 2)], &cycles);

        assert!(fast + deep == trace.len());
        assert_eq!(m1.len(), m2.len());
        assert_eq!(m1[0].mutant_hash, m2[0].mutant_hash);

        let classes: HashSet<&str> = m1.iter().map(|m| m.mutation_class.as_str()).collect();
        assert!(classes.contains("M-Policy"));
        assert!(classes.contains("M-Tool"));
        assert!(classes.contains("M-Context"));
        assert!(classes.contains("M-Complex"));
    }

    #[test]
    fn contract_splicer_rejects_malformed_json() {
        assert!(AgentContractSplicer::splice_and_validate_json("{\"a\":1}").is_some());
        assert!(AgentContractSplicer::splice_and_validate_json("{\"a\":1").is_none());
        assert!(AgentContractSplicer::splice_and_validate_json("not-json").is_none());
    }
}
