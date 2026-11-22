use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmTier {
    /// Modelo local leve (KineCat-1B-100MB / Ollama) para baixa/média complexidade
    LocalKineCat,
    /// Modelo externo de alta capacidade acionado quando a complexidade sistêmica exige
    ExternalComplex,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmEndpointConfig {
    /// Ex.: "127.0.0.1:11434" (Ollama / KineCat local HTTP)
    pub local_host_port: Option<String>,
    pub local_model: String,
    /// Ex.: "127.0.0.1:8080" ou proxy HTTPS local
    pub external_host_port: Option<String>,
    pub external_model: String,
}

impl Default for LlmEndpointConfig {
    fn default() -> Self {
        Self {
            local_host_port: None,
            local_model: "kinecat-1b-100mb".to_string(),
            external_host_port: None,
            external_model: "external-reasoning".to_string(),
        }
    }
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

    /// Invoca endpoint HTTP/1.1 real (Ollama/OpenAI-compatible) quando configurado,
    /// respeitando `BudgetGuard` e retornando `None` se offline/não configurado.
    pub fn invoke_http_probe(
        &self,
        tier: LlmTier,
        cfg: &LlmEndpointConfig,
        prompt: &str,
        budget: &BudgetGuard,
    ) -> Option<String> {
        let (host_port, model) = match tier {
            LlmTier::LocalKineCat => (cfg.local_host_port.as_deref()?, cfg.local_model.as_str()),
            LlmTier::ExternalComplex => {
                (cfg.external_host_port.as_deref()?, cfg.external_model.as_str())
            }
        };
        if !budget.try_consume(256, 20) {
            return None;
        }
        let body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false
        })
        .to_string();
        http_post_json(host_port, "/api/generate", &body)
    }
}

pub(crate) fn http_post_json(host_port: &str, path: &str, json_body: &str) -> Option<String> {
    let mut stream = TcpStream::connect(host_port).ok()?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
    let req = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        path,
        host_port,
        json_body.len(),
        json_body
    );
    stream.write_all(req.as_bytes()).ok()?;
    let mut resp = String::new();
    stream.read_to_string(&mut resp).ok()?;
    resp.split("\r\n\r\n").nth(1).map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn http_probe_connects_to_live_endpoint_and_respects_budget() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
                let reply = "HTTP/1.1 200 OK\r\nContent-Length: 17\r\n\r\n{\"response\":\"ok\"}";
                let _ = stream.write_all(reply.as_bytes());
            }
        });

        let router = HybridLlmRouter::new(0.45);
        let cfg = LlmEndpointConfig {
            local_host_port: Some(addr),
            ..Default::default()
        };
        let budget = BudgetGuard::new(10, 1000, 1.0);
        let res = router.invoke_http_probe(LlmTier::LocalKineCat, &cfg, "probe", &budget);
        assert_eq!(res.as_deref(), Some("{\"response\":\"ok\"}"));
        assert_eq!(budget.tokens_consumed(), 256);
    }
}
