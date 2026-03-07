//! Numbered context registry for delegate output references.
//!
//! When the Orchestrator delegates work to a role, the output is automatically
//! stored with an auto-incrementing numeric ID.  Subsequent delegate calls can
//! reference prior outputs by ID (`context_refs: [1, 3]`) instead of the
//! Orchestrator re-outputting the full text, saving significant output tokens.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Maximum number of stored context entries before oldest are evicted (FIFO).
const MAX_ENTRIES: usize = 128;
/// Maximum character length for a single stored context entry.
const MAX_ENTRY_CHARS: usize = 100_000;

/// A single stored context snapshot.
#[derive(Debug, Clone)]
pub struct ContextEntry {
    /// Auto-incremented numeric ID (starts at 1).
    pub id: u32,
    /// Which role/agent produced this output.
    pub agent: String,
    /// The task/prompt that was given to the agent.
    pub task_summary: String,
    /// The full output text from the agent.
    pub output: String,
    /// Unix timestamp (seconds) when the entry was created.
    pub created_at: u64,
}

/// Thread-safe, auto-incrementing context registry shared between
/// the Orchestrator's `DelegateTool` and any code that needs to
/// resolve context references.
#[derive(Clone)]
pub struct ContextRegistry {
    next_id: Arc<AtomicU32>,
    entries: Arc<RwLock<HashMap<u32, ContextEntry>>>,
    /// Ordered list of IDs for FIFO eviction.
    order: Arc<RwLock<Vec<u32>>>,
}

impl ContextRegistry {
    pub fn new() -> Self {
        Self {
            next_id: Arc::new(AtomicU32::new(1)),
            entries: Arc::new(RwLock::new(HashMap::new())),
            order: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Store a new context entry. Returns the assigned numeric ID.
    pub async fn register(
        &self,
        agent: impl Into<String>,
        task_summary: impl Into<String>,
        output: impl Into<String>,
    ) -> u32 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut output_text = output.into();
        if output_text.len() > MAX_ENTRY_CHARS {
            output_text.truncate(MAX_ENTRY_CHARS);
            output_text.push_str("\n... [truncated]");
        }

        let entry = ContextEntry {
            id,
            agent: agent.into(),
            task_summary: task_summary.into(),
            output: output_text,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let mut entries = self.entries.write().await;
        let mut order = self.order.write().await;

        entries.insert(id, entry);
        order.push(id);

        // FIFO eviction when over capacity
        while order.len() > MAX_ENTRIES {
            if let Some(oldest_id) = order.first().copied() {
                order.remove(0);
                entries.remove(&oldest_id);
            }
        }

        id
    }

    /// Look up a single entry by ID.
    pub async fn get(&self, id: u32) -> Option<ContextEntry> {
        let entries = self.entries.read().await;
        entries.get(&id).cloned()
    }

    /// Resolve multiple IDs into a combined context string.
    /// Returns `(combined_text, missing_ids)`.
    pub async fn resolve_refs(&self, ids: &[u32]) -> (String, Vec<u32>) {
        let entries = self.entries.read().await;
        let mut parts = Vec::new();
        let mut missing = Vec::new();

        for &id in ids {
            match entries.get(&id) {
                Some(entry) => {
                    parts.push(format!(
                        "[Context #{}] (from {} — {})\n{}",
                        entry.id, entry.agent, entry.task_summary, entry.output
                    ));
                }
                None => missing.push(id),
            }
        }

        (parts.join("\n\n"), missing)
    }

    /// List all stored entries (summary only, without full output).
    /// Returns a concise listing Orchestrator can use for orientation.
    pub async fn list_summaries(&self) -> Vec<(u32, String, String, u64)> {
        let entries = self.entries.read().await;
        let order = self.order.read().await;

        order
            .iter()
            .filter_map(|id| {
                entries.get(id).map(|e| {
                    (
                        e.id,
                        e.agent.clone(),
                        if e.task_summary.len() > 120 {
                            format!("{}...", &e.task_summary[..120])
                        } else {
                            e.task_summary.clone()
                        },
                        e.created_at,
                    )
                })
            })
            .collect()
    }

    /// Total number of stored entries.
    pub async fn len(&self) -> usize {
        self.entries.read().await.len()
    }

    /// Whether the registry is empty.
    pub async fn is_empty(&self) -> bool {
        self.entries.read().await.is_empty()
    }
}

impl Default for ContextRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_and_get() {
        let registry = ContextRegistry::new();
        let id1 = registry.register("coder", "implement login", "fn login() {}").await;
        let id2 = registry.register("critic", "review login", "looks good").await;

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);

        let entry = registry.get(1).await.unwrap();
        assert_eq!(entry.agent, "coder");
        assert_eq!(entry.output, "fn login() {}");

        assert!(registry.get(99).await.is_none());
    }

    #[tokio::test]
    async fn test_resolve_refs() {
        let registry = ContextRegistry::new();
        registry.register("architect", "design API", "use REST").await;
        registry.register("coder", "implement API", "fn api() {}").await;
        registry.register("critic", "review API", "LGTM").await;

        let (combined, missing) = registry.resolve_refs(&[1, 3]).await;
        assert!(missing.is_empty());
        assert!(combined.contains("[Context #1]"));
        assert!(combined.contains("use REST"));
        assert!(combined.contains("[Context #3]"));
        assert!(combined.contains("LGTM"));
        // Should NOT contain context #2
        assert!(!combined.contains("fn api()"));
    }

    #[tokio::test]
    async fn test_resolve_refs_with_missing() {
        let registry = ContextRegistry::new();
        registry.register("coder", "task", "output").await;

        let (_, missing) = registry.resolve_refs(&[1, 5, 10]).await;
        assert_eq!(missing, vec![5, 10]);
    }

    #[tokio::test]
    async fn test_fifo_eviction() {
        let registry = ContextRegistry::new();
        // Register MAX_ENTRIES + 5 entries
        for i in 0..(MAX_ENTRIES + 5) {
            registry
                .register("agent", format!("task {i}"), format!("output {i}"))
                .await;
        }

        assert_eq!(registry.len().await, MAX_ENTRIES);
        // First 5 should have been evicted (IDs 1-5)
        assert!(registry.get(1).await.is_none());
        assert!(registry.get(5).await.is_none());
        // Last entry should exist
        assert!(registry.get((MAX_ENTRIES + 5) as u32).await.is_some());
    }

    #[tokio::test]
    async fn test_list_summaries() {
        let registry = ContextRegistry::new();
        registry.register("architect", "design module", "output1").await;
        registry.register("coder", "implement module", "output2").await;

        let summaries = registry.list_summaries().await;
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].0, 1); // id
        assert_eq!(summaries[0].1, "architect"); // agent
        assert_eq!(summaries[1].0, 2);
        assert_eq!(summaries[1].1, "coder");
    }

    #[tokio::test]
    async fn test_truncation() {
        let registry = ContextRegistry::new();
        let long_output = "x".repeat(MAX_ENTRY_CHARS + 1000);
        registry.register("agent", "task", long_output).await;

        let entry = registry.get(1).await.unwrap();
        assert!(entry.output.len() <= MAX_ENTRY_CHARS + 20); // +20 for "... [truncated]"
        assert!(entry.output.ends_with("... [truncated]"));
    }
}
