//! Team shared context tool for inter-role collaboration.
//!
//! Implements the `team_context` tool that allows roles to read, write, and
//! list shared context entries managed by the coordination bus. This enables
//! true team collaboration where roles can share findings, decisions, and
//! intermediate results with each other without relying on the orchestrator
//! to manually pass context between calls.

use super::traits::{Tool, ToolResult};
use crate::coordination::{CoordinationEnvelope, CoordinationPayload, InMemoryMessageBus};
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

/// Maximum context value size in characters.
const MAX_CONTEXT_VALUE_CHARS: usize = 8000;
/// Maximum number of entries to return in a list operation.
const MAX_LIST_ENTRIES: usize = 50;

/// Tool that provides read/write access to the team shared context store.
///
/// The shared context is backed by the coordination bus's `SharedContextEntry`
/// mechanism, which supports versioned writes with optimistic locking. Roles
/// can use this tool to:
/// - **write**: Publish findings, decisions, or intermediate results
/// - **read**: Retrieve context written by other roles
/// - **list**: Browse available context keys
pub struct TeamContextTool {
    bus: InMemoryMessageBus,
    security: Arc<SecurityPolicy>,
    /// The logical identity used as the "from" agent in coordination messages.
    lead_agent: String,
}

impl TeamContextTool {
    pub fn new(
        bus: InMemoryMessageBus,
        security: Arc<SecurityPolicy>,
        lead_agent: impl Into<String>,
    ) -> Self {
        Self {
            bus,
            security,
            lead_agent: lead_agent.into(),
        }
    }
}

#[async_trait]
impl Tool for TeamContextTool {
    fn name(&self) -> &str {
        "team_context"
    }

    fn description(&self) -> &str {
        "Read, write, or list shared team context. Roles use this to share findings, \
         decisions, and intermediate results with each other. Actions: \
         'write' to publish a context entry (key + value), \
         'read' to retrieve a context entry by key, \
         'list' to browse available context keys with optional prefix filter."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["write", "read", "list"],
                    "description": "The operation to perform on the shared context"
                },
                "key": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Context key for read/write. Use descriptive keys like 'architecture/decisions', 'review/findings', 'code/module_x'"
                },
                "value": {
                    "type": "string",
                    "description": "Context value to write (required for 'write' action)"
                },
                "prefix": {
                    "type": "string",
                    "description": "Optional prefix filter for 'list' action (e.g. 'architecture/' to list all architecture context)"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .ok_or_else(|| anyhow::anyhow!("Missing 'action' parameter"))?;

        match action {
            "write" => self.execute_write(&args),
            "read" => self.execute_read(&args),
            "list" => self.execute_list(&args),
            other => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Unknown action '{other}'. Must be one of: write, read, list"
                )),
            }),
        }
    }
}

impl TeamContextTool {
    fn execute_write(&self, args: &serde_json::Value) -> anyhow::Result<ToolResult> {
        // Security: writing shared context is an act operation
        if let Err(error) = self
            .security
            .enforce_tool_operation(crate::security::policy::ToolOperation::Act, "team_context")
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error),
            });
        }

        let key = args
            .get("key")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing 'key' parameter for write action"))?;

        let value = args
            .get("value")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing 'value' parameter for write action"))?;

        if value.len() > MAX_CONTEXT_VALUE_CHARS {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Context value too large ({} chars, max {MAX_CONTEXT_VALUE_CHARS})",
                    value.len()
                )),
            });
        }

        // Prefix team context keys to distinguish from delegate coordination entries
        let context_key = format!("team/{key}");

        // Determine expected version for optimistic locking
        let expected_version = self
            .bus
            .context_entry(&context_key)
            .map(|e| e.version)
            .unwrap_or(0);

        let mut envelope = CoordinationEnvelope::new_broadcast(
            self.lead_agent.clone(),
            "team-context",
            "team.context",
            CoordinationPayload::ContextPatch {
                key: context_key.clone(),
                expected_version,
                value: serde_json::Value::String(value.to_string()),
            },
        );
        envelope.correlation_id = Some(format!("team-ctx-{}", uuid::Uuid::new_v4()));

        match self.bus.publish(envelope) {
            Ok(_receipt) => Ok(ToolResult {
                success: true,
                output: json!({
                    "action": "write",
                    "key": key,
                    "version": expected_version + 1,
                    "message": format!("Context entry '{key}' written successfully (version {})", expected_version + 1)
                })
                .to_string(),
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to write context entry '{key}': {e}")),
            }),
        }
    }

    fn execute_read(&self, args: &serde_json::Value) -> anyhow::Result<ToolResult> {
        let key = args
            .get("key")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing 'key' parameter for read action"))?;

        // Try with team/ prefix first, then try raw key
        let context_key = format!("team/{key}");
        let entry = self
            .bus
            .context_entry(&context_key)
            .or_else(|| self.bus.context_entry(key));

        match entry {
            Some(entry) => {
                let value_str = match &entry.value {
                    serde_json::Value::String(s) => s.clone(),
                    other => serde_json::to_string_pretty(other).unwrap_or_default(),
                };
                Ok(ToolResult {
                    success: true,
                    output: json!({
                        "action": "read",
                        "key": key,
                        "value": value_str,
                        "version": entry.version,
                        "updated_by": entry.updated_by,
                    })
                    .to_string(),
                    error: None,
                })
            }
            None => Ok(ToolResult {
                success: true,
                output: json!({
                    "action": "read",
                    "key": key,
                    "found": false,
                    "message": format!("No context entry found for key '{key}'")
                })
                .to_string(),
                error: None,
            }),
        }
    }

    fn execute_list(&self, args: &serde_json::Value) -> anyhow::Result<ToolResult> {
        let prefix = args
            .get("prefix")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .unwrap_or("");

        let all_entries = self.bus.context_entries_recent(0);

        // Filter by prefix (check both "team/" prefixed and raw key)
        let team_prefix = if prefix.is_empty() {
            "team/".to_string()
        } else {
            format!("team/{prefix}")
        };

        let delegate_prefix = "delegate/";

        let mut entries: Vec<serde_json::Value> = all_entries
            .iter()
            .filter(|(key, _)| {
                if prefix.is_empty() {
                    // Show team context and delegate output, not internal state
                    key.starts_with("team/")
                        || (key.starts_with(delegate_prefix)
                            && (key.contains("/output") || key.contains("/result")))
                } else {
                    key.starts_with(&team_prefix)
                        || key.starts_with(prefix)
                        || key.contains(prefix)
                }
            })
            .take(MAX_LIST_ENTRIES)
            .map(|(key, entry)| {
                // Strip "team/" prefix for display
                let display_key = key.strip_prefix("team/").unwrap_or(key);
                let value_preview = match &entry.value {
                    serde_json::Value::String(s) => {
                        if s.len() > 200 {
                            format!("{}...", &s[..200])
                        } else {
                            s.clone()
                        }
                    }
                    other => {
                        let s = serde_json::to_string(other).unwrap_or_default();
                        if s.len() > 200 {
                            format!("{}...", &s[..200])
                        } else {
                            s
                        }
                    }
                };
                json!({
                    "key": display_key,
                    "version": entry.version,
                    "updated_by": entry.updated_by,
                    "preview": value_preview,
                })
            })
            .collect();

        entries.sort_by(|a, b| {
            a["key"]
                .as_str()
                .unwrap_or("")
                .cmp(b["key"].as_str().unwrap_or(""))
        });

        Ok(ToolResult {
            success: true,
            output: json!({
                "action": "list",
                "prefix": prefix,
                "count": entries.len(),
                "entries": entries,
            })
            .to_string(),
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_security() -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy::default())
    }

    fn make_bus() -> InMemoryMessageBus {
        let bus = InMemoryMessageBus::new();
        bus.register_agent("lead").expect("register lead");
        bus.register_agent("architect").expect("register architect");
        bus.register_agent("coder").expect("register coder");
        bus
    }

    fn make_tool(bus: InMemoryMessageBus) -> TeamContextTool {
        TeamContextTool::new(bus, test_security(), "lead")
    }

    #[test]
    fn name_and_schema() {
        let bus = make_bus();
        let tool = make_tool(bus);
        assert_eq!(tool.name(), "team_context");
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["action"].is_object());
        assert!(schema["properties"]["key"].is_object());
        assert!(schema["properties"]["value"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("action")));
    }

    #[tokio::test]
    async fn write_and_read_context() {
        let bus = make_bus();
        let tool = make_tool(bus);

        // Write
        let result = tool
            .execute(json!({
                "action": "write",
                "key": "architecture/decisions",
                "value": "Use microservices architecture with gRPC"
            }))
            .await
            .unwrap();
        assert!(result.success, "write failed: {:?}", result.error);

        let output: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output["action"], "write");
        assert_eq!(output["key"], "architecture/decisions");
        assert_eq!(output["version"], 1);

        // Read
        let result = tool
            .execute(json!({
                "action": "read",
                "key": "architecture/decisions"
            }))
            .await
            .unwrap();
        assert!(result.success, "read failed: {:?}", result.error);

        let output: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output["action"], "read");
        assert_eq!(output["value"], "Use microservices architecture with gRPC");
        assert_eq!(output["version"], 1);
    }

    #[tokio::test]
    async fn read_nonexistent_key() {
        let bus = make_bus();
        let tool = make_tool(bus);

        let result = tool
            .execute(json!({
                "action": "read",
                "key": "nonexistent/key"
            }))
            .await
            .unwrap();
        assert!(result.success);

        let output: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output["found"], false);
    }

    #[tokio::test]
    async fn list_context_entries() {
        let bus = make_bus();
        let tool = make_tool(bus);

        // Write multiple entries
        tool.execute(json!({
            "action": "write",
            "key": "arch/decisions",
            "value": "Decision 1"
        }))
        .await
        .unwrap();
        tool.execute(json!({
            "action": "write",
            "key": "arch/trade-offs",
            "value": "Trade-off 1"
        }))
        .await
        .unwrap();
        tool.execute(json!({
            "action": "write",
            "key": "code/module_a",
            "value": "Module A implementation"
        }))
        .await
        .unwrap();

        // List all
        let result = tool
            .execute(json!({"action": "list"}))
            .await
            .unwrap();
        assert!(result.success);
        let output: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output["count"], 3);

        // List with prefix filter
        let result = tool
            .execute(json!({"action": "list", "prefix": "arch/"}))
            .await
            .unwrap();
        assert!(result.success);
        let output: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output["count"], 2);
    }

    #[tokio::test]
    async fn write_version_increments() {
        let bus = make_bus();
        let tool = make_tool(bus);

        // First write
        let result = tool
            .execute(json!({
                "action": "write",
                "key": "test/versioned",
                "value": "v1"
            }))
            .await
            .unwrap();
        let output: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output["version"], 1);

        // Second write to same key
        let result = tool
            .execute(json!({
                "action": "write",
                "key": "test/versioned",
                "value": "v2"
            }))
            .await
            .unwrap();
        assert!(result.success, "write v2 failed: {:?}", result.error);
        let output: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output["version"], 2);
    }

    #[tokio::test]
    async fn invalid_action_rejected() {
        let bus = make_bus();
        let tool = make_tool(bus);

        let result = tool
            .execute(json!({"action": "delete"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Unknown action"));
    }

    #[tokio::test]
    async fn write_without_key_rejected() {
        let bus = make_bus();
        let tool = make_tool(bus);

        let result = tool
            .execute(json!({"action": "write", "value": "test"}))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn write_without_value_rejected() {
        let bus = make_bus();
        let tool = make_tool(bus);

        let result = tool
            .execute(json!({"action": "write", "key": "test"}))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn value_size_limit_enforced() {
        let bus = make_bus();
        let tool = make_tool(bus);

        let large_value = "x".repeat(MAX_CONTEXT_VALUE_CHARS + 1);
        let result = tool
            .execute(json!({
                "action": "write",
                "key": "test/large",
                "value": large_value
            }))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("too large"));
    }

    #[tokio::test]
    async fn readonly_blocks_write() {
        let bus = make_bus();
        let security = Arc::new(SecurityPolicy {
            autonomy: crate::security::AutonomyLevel::ReadOnly,
            ..SecurityPolicy::default()
        });
        let tool = TeamContextTool::new(bus, security, "lead");

        let result = tool
            .execute(json!({
                "action": "write",
                "key": "test",
                "value": "blocked"
            }))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("read-only mode"));
    }
}
