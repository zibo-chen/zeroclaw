//! LLM Debug Logger — captures full request/response payloads for context engineering analysis.
//!
//! Writes structured JSONL entries to `~/.coraldesk/debug/llm_calls.jsonl`.
//! Each entry contains the full messages array sent to the LLM and the complete response,
//! enabling developers to inspect and optimize their context engineering.

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, RwLock};
use uuid::Uuid;

use crate::providers::ChatMessage;

const DEFAULT_DEBUG_DIR: &str = ".coraldesk/debug";
const DEBUG_FILENAME: &str = "llm_calls.jsonl";
const MAX_ENTRIES: usize = 500;

/// Global flag to enable/disable LLM debug logging.
static LLM_DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

/// A single LLM call debug entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmDebugEntry {
    /// Unique ID for this entry.
    pub id: String,
    /// ISO-8601 timestamp.
    pub timestamp: String,
    /// Session ID (if applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Provider name (e.g. "openrouter", "anthropic").
    pub provider: String,
    /// Model name (e.g. "anthropic/claude-sonnet-4-20250514").
    pub model: String,
    /// Temperature setting.
    pub temperature: f64,
    /// Tool call loop iteration (0-based).
    pub iteration: usize,
    /// The full messages array sent to the LLM.
    pub request_messages: Vec<LlmDebugMessage>,
    /// Tool specs sent with the request (names only for brevity).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_names: Option<Vec<String>>,
    /// The LLM response text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_text: Option<String>,
    /// Tool calls in the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_tool_calls: Option<Vec<LlmDebugToolCall>>,
    /// Input token count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Output token count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Request duration in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u128>,
    /// Whether the call succeeded.
    pub success: bool,
    /// Error message if failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Stop reason from the LLM.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

/// A message in the request messages array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmDebugMessage {
    pub role: String,
    pub content: String,
    /// Approximate character count (useful for sizing analysis).
    pub char_count: usize,
}

/// A tool call from the LLM response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmDebugToolCall {
    pub name: String,
    pub arguments: String,
}

impl LlmDebugMessage {
    pub fn from_chat_message(msg: &ChatMessage) -> Self {
        Self {
            role: msg.role.clone(),
            content: msg.content.clone(),
            char_count: msg.content.len(),
        }
    }
}

/// LLM debug logger that writes to a JSONL file.
struct LlmDebugLogger {
    path: PathBuf,
    write_lock: std::sync::Mutex<()>,
}

impl LlmDebugLogger {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            write_lock: std::sync::Mutex::new(()),
        }
    }

    fn append(&self, entry: &LlmDebugEntry) -> Result<()> {
        let _guard = self.write_lock.lock().unwrap_or_else(|e| e.into_inner());

        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }

        let line = serde_json::to_string(entry)?;
        let mut options = OpenOptions::new();
        options.create(true).append(true);

        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }

        let mut file = options.open(&self.path)?;
        writeln!(file, "{line}")?;
        file.sync_data()?;

        // Trim to max entries to prevent unbounded growth
        self.trim_if_needed()?;

        Ok(())
    }

    fn trim_if_needed(&self) -> Result<()> {
        let raw = fs::read_to_string(&self.path).unwrap_or_default();
        let lines: Vec<&str> = raw.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.len() <= MAX_ENTRIES {
            return Ok(());
        }
        let keep_from = lines.len().saturating_sub(MAX_ENTRIES);
        let kept = &lines[keep_from..];
        let mut rewritten = kept.join("\n");
        rewritten.push('\n');
        fs::write(&self.path, rewritten)?;
        Ok(())
    }

    fn load_entries(
        &self,
        limit: usize,
        session_filter: Option<&str>,
    ) -> Result<Vec<LlmDebugEntry>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let raw = fs::read_to_string(&self.path)?;
        let mut entries = Vec::new();
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<LlmDebugEntry>(trimmed) {
                Ok(entry) => entries.push(entry),
                Err(err) => tracing::warn!("Skipping malformed LLM debug entry: {err}"),
            }
        }

        // Apply session filter
        if let Some(sid) = session_filter {
            entries.retain(|e| e.session_id.as_deref() == Some(sid));
        }

        // Return most recent first, limited
        if entries.len() > limit {
            let keep_from = entries.len() - limit;
            entries = entries.split_off(keep_from);
        }
        entries.reverse();
        Ok(entries)
    }

    fn clear(&self) -> Result<()> {
        let _guard = self.write_lock.lock().unwrap_or_else(|e| e.into_inner());
        if self.path.exists() {
            fs::remove_file(&self.path)?;
        }
        Ok(())
    }

    fn get_path(&self) -> &Path {
        &self.path
    }
}

static DEBUG_LOGGER: LazyLock<RwLock<Option<Arc<LlmDebugLogger>>>> =
    LazyLock::new(|| RwLock::new(None));

/// Resolve the debug log file path.
fn resolve_debug_path() -> PathBuf {
    directories::UserDirs::new()
        .map(|u| u.home_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
        .join(DEFAULT_DEBUG_DIR)
        .join(DEBUG_FILENAME)
}

/// Initialize the LLM debug logger.
fn ensure_logger() -> Arc<LlmDebugLogger> {
    {
        let guard = DEBUG_LOGGER.read().unwrap_or_else(|e| e.into_inner());
        if let Some(ref logger) = *guard {
            return Arc::clone(logger);
        }
    }
    let logger = Arc::new(LlmDebugLogger::new(resolve_debug_path()));
    let mut guard = DEBUG_LOGGER.write().unwrap_or_else(|e| e.into_inner());
    *guard = Some(Arc::clone(&logger));
    logger
}

// ── Public API ──────────────────────────────────────────

/// Check if LLM debug logging is enabled.
pub fn is_enabled() -> bool {
    LLM_DEBUG_ENABLED.load(Ordering::Relaxed)
}

/// Enable or disable LLM debug logging.
pub fn set_enabled(enabled: bool) {
    LLM_DEBUG_ENABLED.store(enabled, Ordering::Relaxed);
    if enabled {
        tracing::info!("LLM debug logging enabled");
        // Ensure logger is initialized
        ensure_logger();
    } else {
        tracing::info!("LLM debug logging disabled");
    }
}

/// Record an LLM request (before the call).
/// Returns a debug entry ID that should be passed to `record_response`.
pub fn record_request(
    session_id: Option<&str>,
    provider: &str,
    model: &str,
    temperature: f64,
    iteration: usize,
    messages: &[ChatMessage],
    tool_names: Option<Vec<String>>,
) -> Option<String> {
    if !is_enabled() {
        return None;
    }

    let id = Uuid::new_v4().to_string();
    let entry = LlmDebugEntry {
        id: id.clone(),
        timestamp: Utc::now().to_rfc3339(),
        session_id: session_id.map(str::to_string),
        provider: provider.to_string(),
        model: model.to_string(),
        temperature,
        iteration,
        request_messages: messages
            .iter()
            .map(LlmDebugMessage::from_chat_message)
            .collect(),
        tool_names,
        response_text: None,
        response_tool_calls: None,
        input_tokens: None,
        output_tokens: None,
        duration_ms: None,
        success: false,
        error: None,
        stop_reason: None,
    };

    let logger = ensure_logger();
    if let Err(err) = logger.append(&entry) {
        tracing::warn!("Failed to write LLM debug request entry: {err}");
    }

    Some(id)
}

/// Record the LLM response (after the call completes).
pub fn record_response(
    request_id: Option<&str>,
    session_id: Option<&str>,
    provider: &str,
    model: &str,
    temperature: f64,
    iteration: usize,
    messages: &[ChatMessage],
    tool_names: Option<Vec<String>>,
    response_text: &str,
    response_tool_calls: &[(String, String)], // (name, arguments)
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    duration_ms: u128,
    success: bool,
    error: Option<&str>,
    stop_reason: Option<&str>,
) {
    if !is_enabled() {
        return;
    }

    let entry = LlmDebugEntry {
        id: request_id
            .map(str::to_string)
            .unwrap_or_else(|| Uuid::new_v4().to_string()),
        timestamp: Utc::now().to_rfc3339(),
        session_id: session_id.map(str::to_string),
        provider: provider.to_string(),
        model: model.to_string(),
        temperature,
        iteration,
        request_messages: messages
            .iter()
            .map(LlmDebugMessage::from_chat_message)
            .collect(),
        tool_names,
        response_text: Some(response_text.to_string()),
        response_tool_calls: if response_tool_calls.is_empty() {
            None
        } else {
            Some(
                response_tool_calls
                    .iter()
                    .map(|(name, args)| LlmDebugToolCall {
                        name: name.clone(),
                        arguments: args.clone(),
                    })
                    .collect(),
            )
        },
        input_tokens,
        output_tokens,
        duration_ms: Some(duration_ms),
        success,
        error: error.map(str::to_string),
        stop_reason: stop_reason.map(str::to_string),
    };

    let logger = ensure_logger();
    if let Err(err) = logger.append(&entry) {
        tracing::warn!("Failed to write LLM debug response entry: {err}");
    }
}

/// Load recent debug entries.
pub fn load_entries(limit: usize, session_filter: Option<&str>) -> Result<Vec<LlmDebugEntry>> {
    let logger = ensure_logger();
    logger.load_entries(limit, session_filter)
}

/// Clear all debug entries.
pub fn clear_entries() -> Result<()> {
    let logger = ensure_logger();
    logger.clear()
}

/// Get the debug log file path.
pub fn get_log_path() -> String {
    let logger = ensure_logger();
    logger.get_path().to_string_lossy().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_load() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test_debug.jsonl");
        let logger = Arc::new(LlmDebugLogger::new(path));
        let mut guard = DEBUG_LOGGER.write().unwrap();
        let prev = guard.take();
        *guard = Some(Arc::clone(&logger));
        drop(guard);

        LLM_DEBUG_ENABLED.store(true, Ordering::Relaxed);

        let messages = vec![
            ChatMessage::system("You are a helpful assistant.".to_string()),
            ChatMessage::user("Hello".to_string()),
        ];

        record_response(
            Some("test-id"),
            Some("session-1"),
            "openrouter",
            "claude-sonnet",
            0.7,
            0,
            &messages,
            Some(vec!["file_read".to_string()]),
            "Hello! How can I help?",
            &[],
            Some(100),
            Some(50),
            150,
            true,
            None,
            Some("end_turn"),
        );

        let entries = logger.load_entries(10, None).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "test-id");
        assert_eq!(entries[0].request_messages.len(), 2);
        assert_eq!(entries[0].request_messages[0].role, "system");
        assert_eq!(
            entries[0].response_text.as_deref(),
            Some("Hello! How can I help?")
        );

        // Restore
        LLM_DEBUG_ENABLED.store(false, Ordering::Relaxed);
        let mut guard = DEBUG_LOGGER.write().unwrap();
        *guard = prev;
    }
}
