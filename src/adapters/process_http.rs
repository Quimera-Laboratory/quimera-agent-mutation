use crate::adapters::{AgentAdapter, AgentFrameworkKind, ReplayExecutionResult};
use crate::llm_router::http_post_json;
use crate::quimera_agent_mutator::AgentMutant;
use std::io::Write;
use std::process::{Command, Stdio};

#[derive(Clone, Debug)]
pub enum BridgeTransport {
    /// Simulação segura em memória (default offline)
    DryRunSandbox,
    /// HTTP/1.1 POST para endpoint real (ex.: "127.0.0.1:8000" + "/mutate-eval")
    Http { host_port: String, path: String },
    /// Subprocesso CLI / MCP stdio (recebe JSON do mutante via stdin, retorna ReplayExecutionResult em stdout)
    StdioCommand { program: String, args: Vec<String> },
}

/// Adaptador genérico HTTP/CLI e bridges para LangGraph, CrewAI, Google ADK e OpenAI SDK.
pub struct GenericBridgeAdapter {
    pub kind: AgentFrameworkKind,
    pub endpoint_or_cmd: String,
    pub transport: BridgeTransport,
}

impl GenericBridgeAdapter {
    pub fn new(kind: AgentFrameworkKind, endpoint_or_cmd: impl Into<String>) -> Self {
        Self {
            kind,
            endpoint_or_cmd: endpoint_or_cmd.into(),
            transport: BridgeTransport::DryRunSandbox,
        }
    }

    pub fn with_http(
        kind: AgentFrameworkKind,
        host_port: impl Into<String>,
        path: impl Into<String>,
    ) -> Self {
        let hp = host_port.into();
        let p = path.into();
        Self {
            kind,
            endpoint_or_cmd: format!("http://{}{}", hp, p),
            transport: BridgeTransport::Http {
                host_port: hp,
                path: p,
            },
        }
    }

    pub fn with_stdio(
        kind: AgentFrameworkKind,
        program: impl Into<String>,
        args: Vec<String>,
    ) -> Self {
        let prog = program.into();
        Self {
            kind,
            endpoint_or_cmd: prog.clone(),
            transport: BridgeTransport::StdioCommand {
                program: prog,
                args,
            },
        }
    }

    fn execute_via_stdio(
        &self,
        program: &str,
        args: &[String],
        payload_json: &str,
    ) -> Option<ReplayExecutionResult> {
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(payload_json.as_bytes());
        }
        let output = child.wait_with_output().ok()?;
        if !output.status.success() {
            return None;
        }
        serde_json::from_slice::<ReplayExecutionResult>(&output.stdout).ok()
    }
}

impl AgentAdapter for GenericBridgeAdapter {
    fn framework(&self) -> AgentFrameworkKind {
        self.kind
    }

    fn run_baseline(&self) -> ReplayExecutionResult {
        match &self.transport {
            BridgeTransport::Http { host_port, path } => {
                let body = r#"{"mode":"baseline"}"#;
                if let Some(resp) = http_post_json(host_port, path, body) {
                    if let Ok(parsed) = serde_json::from_str::<ReplayExecutionResult>(&resp) {
                        return parsed;
                    }
                }
            }
            BridgeTransport::StdioCommand { program, args } => {
                if let Some(parsed) = self.execute_via_stdio(program, args, r#"{"mode":"baseline"}"#)
                {
                    return parsed;
                }
            }
            BridgeTransport::DryRunSandbox => {}
        }
        ReplayExecutionResult {
            detected_or_blocked: false,
            violated_invariant: false,
            looped_or_timed_out: false,
            final_output: format!("bridge:{:?}:baseline_ok", self.kind),
            steps_executed: 2,
        }
    }

    fn run_with_mutant(&self, mutant: &AgentMutant) -> ReplayExecutionResult {
        match &self.transport {
            BridgeTransport::Http { host_port, path } => {
                if let Ok(body) = serde_json::to_string(mutant) {
                    if let Some(resp) = http_post_json(host_port, path, &body) {
                        if let Ok(parsed) = serde_json::from_str::<ReplayExecutionResult>(&resp) {
                            return parsed;
                        }
                    }
                }
            }
            BridgeTransport::StdioCommand { program, args } => {
                if let Ok(body) = serde_json::to_string(mutant) {
                    if let Some(parsed) = self.execute_via_stdio(program, args, &body) {
                        return parsed;
                    }
                }
            }
            BridgeTransport::DryRunSandbox => {}
        }
        let detected = mutant.mutated_payload.contains("\"injected_conflict\":true")
            || mutant.mutated_payload.contains("\"truncated\":true");
        ReplayExecutionResult {
            detected_or_blocked: detected,
            violated_invariant: !detected,
            looped_or_timed_out: false,
            final_output: format!("bridge:{:?}:{}", self.kind, mutant.mutation_operator),
            steps_executed: 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quimera_agent_mutator::AgentGermLayer;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn http_bridge_sends_mutant_to_live_server() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 2048];
                let _ = stream.read(&mut buf);
                let json = r#"{"detected_or_blocked":true,"violated_invariant":false,"looped_or_timed_out":false,"final_output":"live_caught","steps_executed":3}"#;
                let reply = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
                    json.len(),
                    json
                );
                let _ = stream.write_all(reply.as_bytes());
            }
        });

        let adapter = GenericBridgeAdapter::with_http(AgentFrameworkKind::LangGraph, addr, "/eval");
        let mutant = AgentMutant {
            mutant_id: 1,
            mutant_hash: "abc".into(),
            target_step: 0,
            germ_layer: AgentGermLayer::PolicyDecision,
            mutation_class: "M-Policy".into(),
            mutation_operator: "BYPASS".into(),
            original_payload: "{}".into(),
            mutated_payload: "{\"require_confirmation\":false}".into(),
            expected_invariant: "safe".into(),
        };
        let res = adapter.run_with_mutant(&mutant);
        assert!(res.detected_or_blocked);
        assert_eq!(res.final_output, "live_caught");
    }
}
