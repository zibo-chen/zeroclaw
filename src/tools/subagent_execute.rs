//! Synchronous sub-agent execution tool.
//!
//! Implements the `subagent_execute` tool that runs a delegate agent
//! synchronously, blocking until the result is available (with timeout).
//! Unlike `subagent_spawn` which returns immediately with a session ID,
//! this tool waits for the sub-agent to complete and returns the result
//! directly to the caller.

use super::agent_load_tracker::AgentLoadTracker;
use super::agent_selection::{select_agent_with_load, AgentSelectionPolicy};
use super::orchestration_settings::load_orchestration_settings;
use super::traits::{Tool, ToolResult};
use crate::config::{DelegateAgentConfig, SubAgentsConfig};
use crate::observability::traits::{Observer, ObserverEvent, ObserverMetric};
use crate::providers::{self, ChatMessage, Provider};
use crate::security::policy::ToolOperation;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// Default timeout for synchronous sub-agent execution.
const EXECUTE_TIMEOUT_SECS: u64 = 300;

/// Tool that executes a delegate agent synchronously, blocking until
/// the result is available. For long-running tasks, prefer `subagent_spawn`.
pub struct SubAgentExecuteTool {
    agents: Arc<HashMap<String, DelegateAgentConfig>>,
    security: Arc<SecurityPolicy>,
    fallback_credential: Option<String>,
    provider_runtime_options: providers::ProviderRuntimeOptions,
    parent_tools: Arc<Vec<Arc<dyn Tool>>>,
    multimodal_config: crate::config::MultimodalConfig,
    subagent_settings: SubAgentsConfig,
    load_tracker: AgentLoadTracker,
    runtime_config_path: Option<PathBuf>,
}

impl SubAgentExecuteTool {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        agents: HashMap<String, DelegateAgentConfig>,
        fallback_credential: Option<String>,
        security: Arc<SecurityPolicy>,
        provider_runtime_options: providers::ProviderRuntimeOptions,
        parent_tools: Arc<Vec<Arc<dyn Tool>>>,
        multimodal_config: crate::config::MultimodalConfig,
        subagents_enabled: bool,
        auto_activate: bool,
        runtime_config_path: Option<PathBuf>,
    ) -> Self {
        let mut subagent_settings = SubAgentsConfig::default();
        subagent_settings.enabled = subagents_enabled;
        subagent_settings.auto_activate = auto_activate;

        Self {
            agents: Arc::new(agents),
            security,
            fallback_credential,
            provider_runtime_options,
            parent_tools,
            multimodal_config,
            subagent_settings,
            load_tracker: AgentLoadTracker::new(),
            runtime_config_path,
        }
    }

    /// Reuse a shared runtime load tracker.
    pub fn with_load_tracker(mut self, load_tracker: AgentLoadTracker) -> Self {
        self.load_tracker = load_tracker;
        self
    }

    fn runtime_subagent_settings(&self) -> SubAgentsConfig {
        let mut settings = self.subagent_settings.clone();
        settings.load_window_secs = settings.load_window_secs.max(1);

        if let Some(path) = self.runtime_config_path.as_deref() {
            match load_orchestration_settings(path) {
                Ok((_teams, subagents)) => {
                    settings = subagents;
                    settings.load_window_secs = settings.load_window_secs.max(1);
                }
                Err(error) => {
                    tracing::debug!(
                        path = %path.display(),
                        "subagent_execute: failed to hot-reload orchestration settings: {error}"
                    );
                }
            }
        }

        settings
    }
}

#[async_trait]
impl Tool for SubAgentExecuteTool {
    fn name(&self) -> &str {
        "subagent_execute"
    }

    fn description(&self) -> &str {
        "Execute a sub-agent synchronously, blocking until the result is available. \
         Use for tasks that require the result before continuing (e.g. code generation, \
         research queries). For long-running background tasks, prefer subagent_spawn instead. \
         `agent` can be omitted or set to `auto` when subagent auto-activation is enabled."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let agent_names: Vec<&str> = self.agents.keys().map(|s: &String| s.as_str()).collect();
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "agent": {
                    "type": "string",
                    "minLength": 1,
                    "description": format!(
                        "Name of the agent to execute. Available: {}",
                        if agent_names.is_empty() {
                            "(none configured)".to_string()
                        } else {
                            agent_names.join(", ")
                        }
                    )
                },
                "task": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The task/prompt to send to the sub-agent"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context to prepend (e.g. relevant code, prior findings)"
                },
                "timeout_secs": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 600,
                    "description": "Maximum seconds to wait for the sub-agent (default: 300)"
                }
            },
            "required": ["task"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let requested_agent = args.get("agent").and_then(|v| v.as_str()).map(str::trim);

        let task = args
            .get("task")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .ok_or_else(|| anyhow::anyhow!("Missing 'task' parameter"))?;

        if task.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("'task' parameter must not be empty".into()),
            });
        }

        let context = args
            .get("context")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .unwrap_or("");

        let timeout_secs = args
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(EXECUTE_TIMEOUT_SECS)
            .clamp(10, 600);

        let subagent_settings = self.runtime_subagent_settings();
        if !subagent_settings.enabled {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(
                    "Subagents are currently disabled. Re-enable with model_routing_config action set_orchestration."
                        .to_string(),
                ),
            });
        }

        // Security enforcement
        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, "subagent_execute")
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error),
            });
        }

        let load_window_secs = u64::try_from(subagent_settings.load_window_secs).unwrap_or(1);
        let load_snapshot = self
            .load_tracker
            .snapshot(Duration::from_secs(load_window_secs.max(1)));
        let selection_policy = AgentSelectionPolicy {
            strategy: subagent_settings.strategy,
            inflight_penalty: subagent_settings.inflight_penalty,
            recent_selection_penalty: subagent_settings.recent_selection_penalty,
            recent_failure_penalty: subagent_settings.recent_failure_penalty,
        };

        let selection = match select_agent_with_load(
            self.agents.as_ref(),
            requested_agent,
            task,
            context,
            subagent_settings.auto_activate,
            None,
            Some(&load_snapshot),
            selection_policy,
        ) {
            Ok(selection) => selection,
            Err(error) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(error.to_string()),
                });
            }
        };
        let agent_name = selection.agent_name.clone();
        let Some(agent_config) = self.agents.get(&agent_name).cloned() else {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Resolved agent '{agent_name}' is unavailable")),
            });
        };

        // Create provider for this agent
        let provider_credential_owned = agent_config
            .api_key
            .clone()
            .or_else(|| self.fallback_credential.clone());
        #[allow(clippy::option_as_ref_deref)]
        let provider_credential = provider_credential_owned.as_ref().map(String::as_str);

        let provider: Box<dyn Provider> = match providers::create_provider_with_options(
            &agent_config.provider,
            provider_credential,
            &self.provider_runtime_options,
        ) {
            Ok(p) => p,
            Err(e) => {
                self.load_tracker.record_failure(&agent_name);
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "Failed to create provider '{}' for agent '{agent_name}': {e}",
                        agent_config.provider
                    )),
                });
            }
        };

        // Build the message
        let full_prompt = if context.is_empty() {
            task.to_string()
        } else {
            format!("[Context]\n{context}\n\n[Task]\n{task}")
        };

        let mut load_lease = self.load_tracker.start(&agent_name);

        // Execute synchronously with timeout
        let result = if agent_config.agentic {
            self.execute_agentic_sync(
                &agent_name,
                &agent_config,
                &*provider,
                &full_prompt,
                timeout_secs,
            )
            .await
        } else {
            self.execute_simple_sync(
                &agent_name,
                &agent_config,
                &*provider,
                &full_prompt,
                timeout_secs,
            )
            .await
        };

        match &result {
            Ok(r) if r.success => load_lease.mark_success(),
            _ => load_lease.mark_failure(),
        }

        result
    }
}

impl SubAgentExecuteTool {
    async fn execute_simple_sync(
        &self,
        agent_name: &str,
        agent_config: &DelegateAgentConfig,
        provider: &dyn Provider,
        full_prompt: &str,
        timeout_secs: u64,
    ) -> anyhow::Result<ToolResult> {
        let temperature = agent_config.temperature.unwrap_or(0.7);

        let result = tokio::time::timeout(
            Duration::from_secs(timeout_secs),
            provider.chat_with_system(
                agent_config.system_prompt.as_deref(),
                full_prompt,
                &agent_config.model,
                temperature,
            ),
        )
        .await;

        match result {
            Ok(Ok(response)) => {
                let rendered = if response.trim().is_empty() {
                    "[Empty response]".to_string()
                } else {
                    response
                };
                Ok(ToolResult {
                    success: true,
                    output: format!(
                        "[Agent '{agent_name}' ({provider}/{model})]\n{rendered}",
                        provider = agent_config.provider,
                        model = agent_config.model
                    ),
                    error: None,
                })
            }
            Ok(Err(e)) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Agent '{agent_name}' failed: {e}")),
            }),
            Err(_) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Agent '{agent_name}' timed out after {timeout_secs}s"
                )),
            }),
        }
    }

    async fn execute_agentic_sync(
        &self,
        agent_name: &str,
        agent_config: &DelegateAgentConfig,
        provider: &dyn Provider,
        full_prompt: &str,
        timeout_secs: u64,
    ) -> anyhow::Result<ToolResult> {
        if agent_config.allowed_tools.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Agent '{agent_name}' has agentic=true but allowed_tools is empty"
                )),
            });
        }

        let allowed = agent_config
            .allowed_tools
            .iter()
            .map(|name| name.trim())
            .filter(|name| !name.is_empty())
            .collect::<std::collections::HashSet<_>>();

        let sub_tools: Vec<Box<dyn Tool>> = self
            .parent_tools
            .iter()
            .filter(|tool| allowed.contains(tool.name()))
            .filter(|tool| {
                if tool.name() == "delegate" {
                    agent_config.allow_nested_delegate
                } else {
                    tool.name() != "subagent_spawn"
                        && tool.name() != "subagent_manage"
                        && tool.name() != "subagent_execute"
                }
            })
            .map(|tool| Box::new(ToolArcRef::new(tool.clone())) as Box<dyn Tool>)
            .collect();

        if sub_tools.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Agent '{agent_name}' has no executable tools after filtering allowlist ({})",
                    agent_config.allowed_tools.join(", ")
                )),
            });
        }

        let temperature = agent_config.temperature.unwrap_or(0.7);
        let mut history = Vec::new();
        if let Some(system_prompt) = agent_config.system_prompt.as_ref() {
            history.push(ChatMessage::system(system_prompt.clone()));
        }
        history.push(ChatMessage::user(full_prompt.to_string()));

        let noop_observer = NoopObserver;

        let result = tokio::time::timeout(
            Duration::from_secs(timeout_secs),
            crate::agent::loop_::run_tool_call_loop(
                provider,
                &mut history,
                &sub_tools,
                &noop_observer,
                &agent_config.provider,
                &agent_config.model,
                temperature,
                true,
                None,
                "subagent_execute",
                &self.multimodal_config,
                agent_config.max_iterations,
                None,
                None,
                None,
                &[],
                None,
            ),
        )
        .await;

        match result {
            Ok(Ok(response)) => {
                let rendered = if response.trim().is_empty() {
                    "[Empty response]".to_string()
                } else {
                    response
                };
                Ok(ToolResult {
                    success: true,
                    output: format!(
                        "[Agent '{agent_name}' ({provider}/{model}, agentic)]\n{rendered}",
                        provider = agent_config.provider,
                        model = agent_config.model
                    ),
                    error: None,
                })
            }
            Ok(Err(e)) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Agent '{agent_name}' failed: {e}")),
            }),
            Err(_) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Agent '{agent_name}' timed out after {timeout_secs}s"
                )),
            }),
        }
    }
}

struct ToolArcRef {
    inner: Arc<dyn Tool>,
}

impl ToolArcRef {
    fn new(inner: Arc<dyn Tool>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl Tool for ToolArcRef {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.inner.parameters_schema()
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.inner.execute(args).await
    }
}

struct NoopObserver;

impl Observer for NoopObserver {
    fn record_event(&self, _event: &ObserverEvent) {}
    fn record_metric(&self, _metric: &ObserverMetric) {}
    fn name(&self) -> &str {
        "noop"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::{AutonomyLevel, SecurityPolicy};
    use tempfile::TempDir;

    fn test_security() -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy::default())
    }

    fn sample_agents() -> HashMap<String, DelegateAgentConfig> {
        let mut agents = HashMap::new();
        agents.insert(
            "researcher".to_string(),
            DelegateAgentConfig {
                provider: "ollama".to_string(),
                model: "llama3".to_string(),
                system_prompt: Some("You are a research assistant.".to_string()),
                api_key: None,
                enabled: true,
                capabilities: vec!["research".to_string()],
                priority: 0,
                temperature: Some(0.3),
                max_depth: 3,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                role_label: None,
                role_color: None,
                role_icon: None,
                is_preset: false,
                allow_nested_delegate: false,
            },
        );
        agents
    }

    fn make_tool(
        agents: HashMap<String, DelegateAgentConfig>,
        security: Arc<SecurityPolicy>,
    ) -> SubAgentExecuteTool {
        SubAgentExecuteTool::new(
            agents,
            None,
            security,
            providers::ProviderRuntimeOptions::default(),
            Arc::new(Vec::new()),
            crate::config::MultimodalConfig::default(),
            true,
            true,
            None,
        )
    }

    #[allow(clippy::fn_params_excessive_bools)]
    fn write_runtime_orchestration_config(
        path: &std::path::Path,
        subagents_enabled: bool,
        subagents_auto_activate: bool,
    ) {
        let contents = format!(
            r#"
default_provider = "openrouter"
default_model = "anthropic/claude-sonnet-4.6"
default_temperature = 0.7

[agent.teams]
enabled = true

[agent.subagents]
enabled = {subagents_enabled}
auto_activate = {subagents_auto_activate}
max_concurrent = 4
"#
        );
        std::fs::write(path, contents).unwrap();
    }

    #[test]
    fn name_and_schema() {
        let tool = make_tool(sample_agents(), test_security());
        assert_eq!(tool.name(), "subagent_execute");
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["agent"].is_object());
        assert!(schema["properties"]["task"].is_object());
        assert!(schema["properties"]["context"].is_object());
        assert!(schema["properties"]["timeout_secs"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("task")));
        assert_eq!(schema["additionalProperties"], json!(false));
    }

    #[test]
    fn description_not_empty() {
        let tool = make_tool(sample_agents(), test_security());
        assert!(!tool.description().is_empty());
    }

    #[tokio::test]
    async fn missing_task_param() {
        let tool = make_tool(sample_agents(), test_security());
        let result = tool.execute(json!({"agent": "researcher"})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn blank_task_rejected() {
        let tool = make_tool(sample_agents(), test_security());
        let result = tool
            .execute(json!({"agent": "researcher", "task": "  "}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("must not be empty"));
    }

    #[tokio::test]
    async fn unknown_agent_returns_error() {
        let tool = make_tool(sample_agents(), test_security());
        let result = tool
            .execute(json!({"agent": "nonexistent", "task": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Unknown agent"));
    }

    #[tokio::test]
    async fn execute_blocked_in_readonly_mode() {
        let readonly = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::ReadOnly,
            ..SecurityPolicy::default()
        });
        let tool = make_tool(sample_agents(), readonly);
        let result = tool
            .execute(json!({"agent": "researcher", "task": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("read-only mode"));
    }

    #[tokio::test]
    async fn execute_blocked_when_disabled() {
        let tmp = TempDir::new().unwrap();
        let config_path = tmp.path().join("config.toml");
        write_runtime_orchestration_config(&config_path, false, true);

        let tool = SubAgentExecuteTool::new(
            sample_agents(),
            None,
            test_security(),
            providers::ProviderRuntimeOptions::default(),
            Arc::new(Vec::new()),
            crate::config::MultimodalConfig::default(),
            true,
            true,
            Some(config_path),
        );

        let result = tool
            .execute(json!({"agent": "researcher", "task": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result
            .error
            .unwrap_or_default()
            .contains("Subagents are currently disabled"));
    }

    #[tokio::test]
    async fn execute_no_agents_configured() {
        let tool = make_tool(HashMap::new(), test_security());
        let result = tool
            .execute(json!({"agent": "any", "task": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result
            .error
            .unwrap_or_default()
            .contains("No delegate agents are configured"));
    }

    #[tokio::test]
    async fn schema_lists_agent_names() {
        let tool = make_tool(sample_agents(), test_security());
        let schema = tool.parameters_schema();
        let desc = schema["properties"]["agent"]["description"]
            .as_str()
            .unwrap();
        assert!(desc.contains("researcher"));
    }
}
