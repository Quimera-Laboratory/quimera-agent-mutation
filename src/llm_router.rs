use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmTier {
    /// Modelo local leve (KineCat-1B-100MB / Ollama) para baixa/média complexidade
    LocalKineCat,
    /// Modelo externo de alta capacidade acionado quando a complexidade sistêmica exige
    ExternalComplex,
}

#[derive(Debug)]
pub struct BudgetGuard {
    pub max_mutants: usize,
    pub max_tokens: u64,
    pub max_cost_usd_micros: u64,
    tokens_used: AtomicU64,
    cost_used_micros: AtomicU64,
}

impl BudgetGuard {
    pub fn new(max_mutants: usize, max_tokens: u64, max_cost_usd: f64) -> Self {
        Self {
            max_mutants,
            max_tokens,
            max_cost_usd_micros: (max_cost_usd * 1_000_000.0).round() as u64,
            tokens_used: AtomicU64::new(0),
            cost_used_micros: AtomicU64::new(0),
        }
    }

    pub fn try_consume(&self, tokens: u64, cost_usd_micros: u64) -> bool {
        let prev_t = self.tokens_used.fetch_add(tokens, Ordering::SeqCst);
        let prev_c = self.cost_used_micros.fetch_add(cost_usd_micros, Ordering::SeqCst);
        if prev_t + tokens > self.max_tokens || prev_c + cost_usd_micros > self.max_cost_usd_micros {
            self.tokens_used.fetch_sub(tokens, Ordering::SeqCst);
            self.cost_used_micros.fetch_sub(cost_usd_micros, Ordering::SeqCst);
            return false;
        }
        true
    }

    pub fn tokens_consumed(&self) -> u64 {
        self.tokens_used.load(Ordering::SeqCst)
    }
}

/// Roteador híbrido adaptativo: usa modelos locais por padrão e escala para externo
/// apenas quando `system_complexity_score` + ambiguidade ultrapassam o limiar.
#[derive(Clone, Debug)]
pub struct HybridLlmRouter {
    pub complexity_escalation_threshold: f32,
}

impl HybridLlmRouter {
    pub fn new(complexity_escalation_threshold: f32) -> Self {
        Self {
            complexity_escalation_threshold,
        }
    }

    pub fn select_tier(&self, system_complexity_score: f32, step_ambiguity: f32) -> LlmTier {
        let effective = (system_complexity_score * 0.65) + (step_ambiguity * 0.35);
        if effective >= self.complexity_escalation_threshold {
            LlmTier::ExternalComplex
        } else {
            LlmTier::LocalKineCat
        }
    }
}
