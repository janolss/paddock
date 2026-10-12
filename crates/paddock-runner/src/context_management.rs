//! Server-side context management (design:
//!
//! The Anthropic `context_management.edits` contract, applied to the raw
//! Anthropic-shaped message array before template render. Pure: this module
//! never touches the engine or the template - callers hand it the messages
//! and a `count` closure (render + tokenize) and get back the edited
//! messages plus the `applied_edits` report the response carries. All of it
//! is opt-in: no `context_management` in the request, nothing here runs, and
//! an overflowing prompt stays the loud error it always was.
//!
//! Strategies (beta `context-management-2025-06-27`, accepted with or
//! without the header - real-API clients always send it, and requiring it
//! adds a failure mode without adding conformance):
//! - `clear_thinking_20251015`: strip thinking blocks from all but the last
//!   N assistant thinking-turns. No trigger - applies when configured.
//! - `clear_tool_uses_20250919`: past a trigger, replace the OLDEST tool
//!   results (and optionally tool inputs) with placeholders, keeping the
//!   most recent `keep` pairs and everything in `exclude_tools`.
//!   `clear_at_least` skips the whole edit when it would not reclaim enough
//!   - pointless prompt-cache invalidation is the thing that knob exists to
//!     prevent (our radix cache has the same economics as Anthropic's).
//! - `compact_20260112` (beta `compact-2026-01-12`): past an input_tokens
//!   trigger, a summarization generation replaces the "compact span" with a
//!   `compaction` content block. This module owns the pure parts - trigger
//!   math, the span/tail split, the resend rewrite, the post-compaction
//!   message build; the two-iteration orchestration lives in the serving
//!   layer (messages.rs), which is the only place that can run a generation.
//!
//! Spec points pinned here because the public docs leave them open (wire
//! shapes are pinned against anthropic-sdk-python 0.120.2 - see
//! - `clear_at_least` unmet => the edit applies nothing this request (rather
//!   than clearing into `keep`); revisit against the SDK if it ever differs.
//! - a cleared tool result is REPLACED (content -> one text block of
//!   placeholder), never removed - templates keep their structure and the
//!   model is told something was there.
//! - re-clearing an already-cleared pair reclaims ~0 tokens, so repeated
//!   requests converge to no-ops via `clear_at_least` naturally.
//! - the compact SPAN is everything before the pending turn: the tail (kept
//!   raw) starts at the last user message holding no tool_result blocks -
//!   the user's current request plus any tool round-trips under it. This
//!   keeps tool_use/tool_result pairs intact and makes the in-request
//!   continuation and the resend render the identical token stream, which is
//!   what turns every post-compaction turn into a radix-cache hit.
//! - compaction produces no applied_edits entry (the SDK's AppliedEdit union
//!   has only the two clear strategies); its report is the compaction block
//!   itself plus usage.iterations.
//! - a compaction block with null content is a FAILED compaction and
//!   round-trips as a no-op (SDK docstring) - we strip the block and keep
//!   the conversation uncompacted.

use serde_json::{Value, json};

/// What a cleared tool result reads as. The model is told the content was
/// removed; the exact wording is ours (Anthropic does not publish theirs).
pub const CLEARED_RESULT: &str = "[Tool result cleared by context management]";

/// Lead-in the model sees before a compaction summary at render time. The
/// summary rides user-voiced inside the pending turn's first user message -
/// the one shape every template family accepts (an assistant-first or
/// doubled-user opening breaks strict-alternation templates like gemma's).
pub const COMPACTION_FRAME: &str =
    "Summary of the earlier conversation (compacted to save context):\n\n";

/// Default summarization instructions (callers replace them verbatim via the
/// edit's `instructions` field). Anthropic's server-side prompt is
/// unpublished; this is modeled on the structure of their SDK's client-side
/// compaction prompt. The pending user turn survives raw in the tail, so the
/// summary only has to carry the PAST.
pub const DEFAULT_COMPACT_INSTRUCTIONS: &str = "\
The conversation above is about to be compacted: everything before the \
current request will be replaced by the summary you write now. Write a \
structured, self-contained summary that lets the conversation resume \
seamlessly: the user's goals and constraints, what has been done or decided \
so far and why, key facts and results worth carrying forward (including \
important tool results), errors already resolved, and what remains open. Be \
concise but complete - err toward keeping anything that would otherwise \
have to be rediscovered. Reply with the summary only.";

#[derive(Debug, Clone, PartialEq)]
pub enum Threshold {
    InputTokens(u64),
    ToolUses(u64),
}

fn parse_threshold(v: &Value, field: &str) -> Result<Threshold, String> {
    let ty = v
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("input_tokens");
    let val = v
        .get("value")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("context_management: {field}.value must be a positive integer"))?;
    match ty {
        "input_tokens" => Ok(Threshold::InputTokens(val)),
        "tool_uses" => Ok(Threshold::ToolUses(val)),
        other => Err(format!(
            "context_management: invalid {field}.type {other:?}"
        )),
    }
}

#[derive(Debug, Clone)]
pub enum Edit {
    ClearThinking {
        /// None = keep all (no-op edit); Some(n) = keep last n thinking turns.
        keep_turns: Option<u64>,
    },
    ClearToolUses {
        trigger: Threshold,
        /// Most recent tool_use/result pairs preserved.
        keep: u64,
        clear_at_least: Option<Threshold>,
        exclude_tools: Vec<String>,
        clear_tool_inputs: bool,
    },
    Compact {
        /// input_tokens only (SDK: BetaInputTokensTriggerParam). Their default
        /// is 150000 with a documented 50k floor; we accept smaller values -
        /// a 32k-window local server needs compaction at ~24k (documented
        /// deviation, a superset the SDKs never notice).
        trigger: u64,
        /// Replaces DEFAULT_COMPACT_INSTRUCTIONS verbatim when set.
        instructions: Option<String>,
        /// Stop with stop_reason "compaction" right after the summary instead
        /// of continuing into the real generation.
        pause: bool,
    },
}

#[derive(Debug, Clone)]
pub struct Config {
    pub edits: Vec<Edit>,
}

/// Parse the request's `context_management` value. Unknown strategies and
/// malformed configs are loud 400s - a control the caller set is never
/// silently ignored.
pub fn parse(v: &Value) -> Result<Config, String> {
    let edits_v = v
        .get("edits")
        .and_then(Value::as_array)
        .ok_or("context_management.edits must be an array")?;
    let mut edits = Vec::new();
    for (i, e) in edits_v.iter().enumerate() {
        let ty = e
            .get("type")
            .and_then(Value::as_str)
            .ok_or("context_management edit needs a type")?;
        match ty {
            "clear_thinking_20251015" => {
                // documented combination rule: clear_thinking must come first
                if i != 0 {
                    return Err(
                        "context_management: clear_thinking_20251015 must be listed first in edits"
                            .into(),
                    );
                }
                let keep_turns = match e.get("keep") {
                    // local default: keep all. The Anthropic per-model default
                    // table is about their models; keep-all is the
                    // cache-friendly default here and matches 4.6+/5 behavior.
                    None => None,
                    Some(Value::String(s)) if s == "all" => None,
                    Some(k) => {
                        let ty = k.get("type").and_then(Value::as_str);
                        if ty != Some("thinking_turns") {
                            return Err(
                                "context_management: clear_thinking keep must be \"all\" or \
                                 {type: \"thinking_turns\", value: N}"
                                    .into(),
                            );
                        }
                        let n = k
                            .get("value")
                            .and_then(Value::as_u64)
                            .filter(|&n| n > 0)
                            .ok_or("context_management: clear_thinking keep.value must be > 0")?;
                        Some(n)
                    }
                };
                edits.push(Edit::ClearThinking { keep_turns });
            }
            "clear_tool_uses_20250919" => {
                let trigger = match e.get("trigger") {
                    Some(t) => parse_threshold(t, "trigger")?,
                    None => Threshold::InputTokens(100_000), // documented default
                };
                let keep = match e.get("keep") {
                    Some(k) => {
                        if k.get("type").and_then(Value::as_str) != Some("tool_uses") {
                            return Err(
                                "context_management: clear_tool_uses keep.type must be \"tool_uses\""
                                    .into(),
                            );
                        }
                        k.get("value")
                            .and_then(Value::as_u64)
                            .ok_or("context_management: keep.value must be an integer")?
                    }
                    None => 3, // documented default
                };
                let clear_at_least = e
                    .get("clear_at_least")
                    .map(|c| parse_threshold(c, "clear_at_least"))
                    .transpose()?;
                let exclude_tools = e
                    .get("exclude_tools")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let clear_tool_inputs = e
                    .get("clear_tool_inputs")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                edits.push(Edit::ClearToolUses {
                    trigger,
                    keep,
                    clear_at_least,
                    exclude_tools,
                    clear_tool_inputs,
                });
            }
            "compact_20260112" => {
                if edits.iter().any(|e| matches!(e, Edit::Compact { .. })) {
                    return Err(
                        "context_management: only one compact_20260112 edit is allowed".into(),
                    );
                }
                let trigger = match e.get("trigger") {
                    None | Some(Value::Null) => 150_000, // documented default
                    Some(t) => match parse_threshold(t, "trigger")? {
                        Threshold::InputTokens(v) => v,
                        Threshold::ToolUses(_) => {
                            return Err("context_management: compact_20260112 trigger must be \
                                        {type: \"input_tokens\", value: N}"
                                .into());
                        }
                    },
                };
                let instructions = match e.get("instructions") {
                    None | Some(Value::Null) => None,
                    Some(Value::String(s)) => Some(s.clone()),
                    Some(_) => {
                        return Err(
                            "context_management: compact_20260112 instructions must be a string"
                                .into(),
                        );
                    }
                };
                let pause = e
                    .get("pause_after_compaction")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                edits.push(Edit::Compact {
                    trigger,
                    instructions,
                    pause,
                });
            }
            other => return Err(format!("context_management: unknown edit type {other:?}")),
        }
    }
    Ok(Config { edits })
}

/// A fired compact_20260112 edit, for the serving layer to orchestrate (the
/// pure core cannot run the summarization generation).
pub struct CompactFire {
    pub instructions: Option<String>,
    pub pause: bool,
}

/// One applied edit for the response report + the running token count after it.
pub struct Applied {
    /// `applied_edits` entries, Anthropic shape. Compaction never appears
    /// here (SDK pin: the AppliedEdit union has only the clear strategies).
    pub edits: Vec<Value>,
    /// Token count of the final (possibly edited) prompt.
    pub final_tokens: usize,
    /// Set when a compact_20260112 edit's trigger fired over a non-empty
    /// span; the caller runs the summarization pass.
    pub compact: Option<CompactFire>,
}

/// The compact span/tail split: the tail (kept raw) starts at the last user
/// message holding no tool_result blocks - the pending request plus any tool
/// round-trips under it. Everything before is the span a compaction
/// summarizes away. 0 = no span, nothing to compact.
pub fn compact_tail_start(messages: &[Value]) -> usize {
    messages
        .iter()
        .rposition(|m| m.get("role").and_then(Value::as_str) == Some("user") && !has_tool_result(m))
        .unwrap_or(0)
}

fn has_tool_result(m: &Value) -> bool {
    m.get("content")
        .and_then(Value::as_array)
        .is_some_and(|blocks| {
            blocks
                .iter()
                .any(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
        })
}

/// Prepend the framed summary into the conversation's first message. The
/// summary rides user-voiced inside the tail's first user message so every
/// template family renders it (see COMPACTION_FRAME); a tail that does not
/// open with a plain user message (hand-crafted resends) gets a fresh user
/// message instead.
fn prepend_summary(out: &mut Vec<Value>, summary: &str) {
    let text = json!({"type": "text", "text": format!("{COMPACTION_FRAME}{summary}")});
    let user_first = out
        .first()
        .and_then(|m| m.get("role"))
        .and_then(Value::as_str)
        == Some("user");
    if !user_first {
        out.insert(0, json!({"role": "user", "content": [text]}));
        return;
    }
    match out[0].get_mut("content") {
        Some(Value::Array(blocks)) => blocks.insert(0, text),
        Some(v) => {
            // string content (the common plain-message shape): wrap both
            // into text blocks so nothing about the original is lost
            let orig = v.take();
            *v = match orig {
                Value::String(s) => json!([text, {"type": "text", "text": s}]),
                _ => json!([text]),
            };
        }
        None => out[0]["content"] = json!([text]),
    }
}

/// The post-compaction conversation the real generation runs over:
/// [tail with the framed summary prepended]. A later resend of the response
/// (compaction block + generated blocks appended by the client) rewrites to
/// the identical prefix via `resend_rewrite`, so every subsequent turn is a
/// radix-cache hit over this exact token stream.
pub fn compacted_messages(messages: &[Value], summary: &str) -> Vec<Value> {
    let tail_start = compact_tail_start(messages);
    let mut out = messages[tail_start..].to_vec();
    prepend_summary(&mut out, summary);
    out
}

/// Render-time semantics of a round-tripped `compaction` block (active on
/// every request, config or not - a request carrying our own response's
/// content must always be valid). Everything before the last compaction
/// block collapses into its summary: drop the messages before the pending
/// turn that preceded it, drop blocks before the block inside its message
/// (spec: "resend drops blocks before compaction"), and prepend the framed
/// summary exactly as `compacted_messages` did in-request. A null-content
/// block is a FAILED compaction and is a no-op: stripped, nothing dropped.
/// Returns None when there is no compaction block (the common case, borrow
/// the original).
pub fn resend_rewrite(messages: &[Value]) -> Option<Vec<Value>> {
    let mut pos: Option<(usize, usize, Option<String>)> = None;
    for (mi, m) in messages.iter().enumerate() {
        if let Some(blocks) = m.get("content").and_then(Value::as_array) {
            for (bi, b) in blocks.iter().enumerate() {
                if b.get("type").and_then(Value::as_str) == Some("compaction") {
                    pos = Some((
                        mi,
                        bi,
                        b.get("content").and_then(Value::as_str).map(str::to_owned),
                    ));
                }
            }
        }
    }
    let (k, j, summary) = pos?;
    let mut out: Vec<Value>;
    match summary.filter(|s| !s.trim().is_empty()) {
        Some(s) => {
            let tail_start = compact_tail_start(&messages[..k]);
            out = messages[tail_start..].to_vec();
            let k_local = k - tail_start;
            if let Some(blocks) = out[k_local]
                .get_mut("content")
                .and_then(Value::as_array_mut)
            {
                blocks.drain(..=j);
            }
            strip_compaction_blocks(&mut out);
            prepend_summary(&mut out, &s);
        }
        None => {
            out = messages.to_vec();
            strip_compaction_blocks(&mut out);
        }
    }
    Some(out)
}

/// Remove every compaction block (and any message emptied by that). After a
/// rewrite the surviving ones are stale no-ops, and the message converter
/// rejects unknown block types loudly.
fn strip_compaction_blocks(out: &mut Vec<Value>) {
    for m in out.iter_mut() {
        if let Some(blocks) = m.get_mut("content").and_then(Value::as_array_mut) {
            blocks.retain(|b| b.get("type").and_then(Value::as_str) != Some("compaction"));
        }
    }
    out.retain(|m| {
        m.get("content")
            .and_then(Value::as_array)
            .is_none_or(|blocks| !blocks.is_empty())
    });
}

// ── OpenAI Responses dialect (item-level) ───────────────────────────────────
//
// Same algebra as the Anthropic half, applied to Responses INPUT ITEMS before
// they are converted to chat messages. Wire shapes pinned against openai
// 2.53.0:
// - config: `context_management: [{"type": "compaction", "compact_threshold"}]`
//   (compaction is the only entry type; no instructions/pause knobs).
// - the compaction item is `{"id", "type": "compaction", "encrypted_content"}`
//   with encrypted_content required - ours carries plain JSON there (the v2
//   state, `OaState`, below; first-cut items carried a plaintext summary) so
//   unmodified SDK types round-trip it (no encryption theater on a local
//   box; documented deviation).
// - `{"type": "compaction_trigger"}` must be the FINAL input item and forces
//   a compact-now (the whole conversation, no tail split - the caller is
//   archiving, not asking).
// - an empty encrypted_content is the failed-compaction no-op (the twin of
//   the Anthropic null-content pin): stripped, nothing dropped.

/// Absent/null `compact_threshold` default. OpenAI does not publish theirs;
/// mirroring the Anthropic-dialect default keeps the two dialects symmetric,
/// and (same documented deviation) any positive value is accepted.
pub const OA_DEFAULT_THRESHOLD: u64 = 150_000;

/// Parse the Responses `context_management` array. Returns the compaction
/// threshold when a compaction entry is configured.
pub fn oa_parse(entries: &[Value]) -> Result<Option<u64>, String> {
    let mut threshold = None;
    for e in entries {
        match e.get("type").and_then(Value::as_str) {
            Some("compaction") => {
                if threshold.is_some() {
                    return Err("context_management: only one compaction entry is allowed".into());
                }
                threshold = Some(match e.get("compact_threshold") {
                    None | Some(Value::Null) => OA_DEFAULT_THRESHOLD,
                    Some(v) => v.as_u64().filter(|&n| n > 0).ok_or(
                        "context_management: compact_threshold must be a positive integer",
                    )?,
                });
            }
            other => {
                return Err(format!(
                    "context_management: unknown entry type {:?} (only \"compaction\" is served)",
                    other.unwrap_or("missing")
                ));
            }
        }
    }
    Ok(threshold)
}

/// A plain user message item (`type` defaults to "message" on the wire).
/// pub(crate): the standalone compact endpoint echoes exactly these items.
pub(crate) fn oa_user_message(it: &Value) -> bool {
    matches!(
        it.get("type").and_then(Value::as_str),
        None | Some("message")
    ) && it.get("role").and_then(Value::as_str) == Some("user")
}

/// Item-level twin of `compact_tail_start`: the tail starts at the last user
/// message item - the pending request plus its function_call/output
/// round-trips (tool results are separate items in this dialect, so every
/// user message qualifies). 0 = no span, nothing to compact.
pub fn oa_tail_start(items: &[Value]) -> usize {
    items.iter().rposition(oa_user_message).unwrap_or(0)
}

/// Prepend the framed summary into the first item, user-voiced (same
/// template-safety reasoning as `prepend_summary`). The trailing blank line
/// matters: the Responses converter flattens content parts with no
/// separator, so without it the summary would glue onto the user's text.
fn oa_prepend_summary(out: &mut Vec<Value>, summary: &str) {
    let text = json!({"type": "input_text", "text": format!("{COMPACTION_FRAME}{summary}\n\n")});
    if !out.first().is_some_and(oa_user_message) {
        out.insert(
            0,
            json!({"type": "message", "role": "user", "content": [text]}),
        );
        return;
    }
    match out[0].get_mut("content") {
        Some(Value::Array(parts)) => parts.insert(0, text),
        Some(v) => {
            let orig = v.take();
            *v = match orig {
                Value::String(s) => json!([text, {"type": "input_text", "text": s}]),
                _ => json!([text]),
            };
        }
        None => out[0]["content"] = json!([text]),
    }
}

/// The post-compaction item list a first-cut (plaintext) item's request
/// ran over: tail + framed summary. New items are v2 states (`OaState`);
/// this stays as the reference the legacy resend rewrite is tested against.
#[cfg(test)]
fn oa_compacted_items(items: &[Value], summary: &str) -> Vec<Value> {
    let mut out = items[oa_tail_start(items)..].to_vec();
    oa_prepend_summary(&mut out, summary);
    out
}

/// Render-time semantics of a round-tripped compaction ITEM (active on every
/// request, config or not). A v2 state renders the window by itself (see
/// `OaState`). A legacy plaintext item: everything before its tail
/// collapses into its summary; the item itself (and stale earlier ones) is
/// consumed. An empty/missing encrypted_content is a failed compaction:
/// strip-only no-op. None = no compaction item (borrow the original).
pub fn oa_resend_rewrite(items: &[Value]) -> Option<Vec<Value>> {
    let k = items
        .iter()
        .rposition(|it| it.get("type").and_then(Value::as_str) == Some("compaction"))?;
    // a v2 state renders the window by itself (see OaState)
    if let Some(st) = items[k]
        .get("encrypted_content")
        .and_then(Value::as_str)
        .and_then(OaState::decode)
    {
        return Some(oa_render(&st, &items[k + 1..]));
    }
    let summary = items[k]
        .get("encrypted_content")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let mut out: Vec<Value> = match &summary {
        Some(_) => {
            let tail_start = oa_tail_start(&items[..k]);
            items[tail_start..].to_vec()
        }
        None => items.to_vec(),
    };
    out.retain(|it| it.get("type").and_then(Value::as_str) != Some("compaction"));
    if let Some(s) = &summary {
        oa_prepend_summary(&mut out, s);
    }
    Some(out)
}

/// Extract a `compaction_trigger` input item. The SDK pins it as the final
/// input item; anywhere else is a loud error. Returns true (with the item
/// removed) when the caller should run a compact-now.
pub fn oa_take_trigger(items: &mut Vec<Value>) -> Result<bool, String> {
    let Some(pos) = items
        .iter()
        .position(|it| it.get("type").and_then(Value::as_str) == Some("compaction_trigger"))
    else {
        return Ok(false);
    };
    if pos != items.len() - 1 {
        return Err("a compaction_trigger must be the final input item".into());
    }
    items.pop();
    Ok(true)
}

// -- OpenAI Responses compaction state (v2) ----------------------------------
//
// OpenAI's contract for server-side compaction: when the rendered context
// crosses compact_threshold the server compacts and continues, leading the
// output with a compaction item, and "you can drop items that came before
// the most recent compaction item" - the item stands for everything before
// it. Their /compact keeps "all prior user messages verbatim" and replaces
// the assistant messages, reasoning, tool calls and tool results with the
// one item. Agent loops are the case the feature exists for: one user
// request, hundreds of tool calls, and the window fills INSIDE that turn.
//
// The first cut (the plaintext summary above, `oa_compacted_items`) only
// summarized what came before the pending user item and kept that turn raw
// in the CLIENT's copy, so (1) a single-request agent loop could never
// compact - the span before its only user item is empty - and ran into the
// window, and (2) a client that pruned as documented dropped the pending
// request and its tool rounds, which the summary never saw. The v2 item
// therefore carries the window itself: system/developer items and every
// user message verbatim, the summary, and the raw tail it kept (the latest
// tool round, so an agent does not lose the file it just read, or a
// pending user turn). A resend renders from the item alone, pruned or not,
// so the compacting request and every later one render the identical
// prefix - the radix-cache invariant. Legacy plaintext items keep their
// old rewrite.

/// Marker and version of a compaction state inside `encrypted_content`.
const OA_STATE_KEY: &str = "paddock_compaction";
const OA_STATE_VERSION: u64 = 2;

/// Lead-in of the summary in a compacted window. It follows the user's own
/// messages (kept verbatim above it), so it speaks about everything else.
pub const OA_COMPACTION_FRAME: &str = "The rest of this conversation so far - replies, reasoning, \
tool calls and their results - was compacted to save context. Summary of it:\n\n";

/// Labels around the kept user messages in a compacted head: the earlier
/// requests above the summary, the one being worked on below it. Without
/// them a head of several verbatim requests reads as one list, and a model
/// answered the FIRST of them after a compaction (live, 2026-10-12).
pub const OA_EARLIER_FRAME: &str = "Earlier requests in this conversation (verbatim):\n\n";
pub const OA_CURRENT_FRAME: &str = "The current request (verbatim):\n\n";

/// Summarization instructions for in-create compaction (the latest tool
/// round stays raw after the summary).
pub const OA_COMPACT_INSTRUCTIONS: &str = "\
The conversation above is about to be compacted to save context. The user's \
own messages are kept verbatim and the most recent tool round stays as it \
is; everything else - your replies and reasoning, earlier tool calls and \
their results, and any earlier summary - will be replaced by the summary you \
write now. Write a structured, self-contained summary that lets the work \
continue seamlessly: the user's goals and constraints, what has been done so \
far and why, the key facts and results worth carrying forward (names, \
values, findings, decisions - exactly as they appeared), errors already \
resolved, and precisely what remains to be done next. Be concise but \
complete - err toward keeping anything that would otherwise have to be \
rediscovered. Reply with the summary only.";

/// Summarization instructions for an archive (compaction_trigger and the
/// standalone /compact endpoint): no raw tail survives.
pub const OA_COMPACT_ALL_INSTRUCTIONS: &str = "\
The conversation above is about to be compacted to save context. The user's \
own messages are kept verbatim; everything else - your replies and \
reasoning, tool calls and their results, and any earlier summary - will be \
replaced by the summary you write now. Write a structured, self-contained \
summary that lets the conversation resume seamlessly: the user's goals and \
constraints, what has been done so far and why, the key facts and results \
worth carrying forward (names, values, findings, decisions - exactly as they \
appeared), errors already resolved, and what remains open. Be concise but \
complete - err toward keeping anything that would otherwise have to be \
rediscovered. Reply with the summary only.";

/// What a v2 compaction item carries: the window before it, rendered
/// back by [`oa_render`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OaState {
    /// system/developer message items, verbatim (they render first)
    pub system: Vec<Value>,
    /// every user message item the summary replaced the context of,
    /// verbatim and in order
    pub users: Vec<Value>,
    pub summary: String,
    /// raw items after the summary: the current request (when the cut fell
    /// mid-turn) and the latest tool round, or the pending user turn
    pub tail: Vec<Value>,
}

impl OaState {
    /// The `encrypted_content` string (plain JSON: no encryption theater on
    /// a local box - the state is the caller's own conversation).
    pub fn encode(&self) -> String {
        json!({OA_STATE_KEY: OA_STATE_VERSION, "system": self.system, "users": self.users,
               "summary": self.summary, "tail": self.tail})
        .to_string()
    }

    /// A v2 state, or None (a legacy plaintext summary, a failed empty
    /// item, anything else).
    pub fn decode(s: &str) -> Option<Self> {
        let v: Value = serde_json::from_str(s.trim()).ok()?;
        if v.get(OA_STATE_KEY)?.as_u64()? != OA_STATE_VERSION {
            return None;
        }
        let list = |k: &str| {
            v.get(k)
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        };
        let summary = v.get("summary")?.as_str()?.trim().to_owned();
        (!summary.is_empty()).then(|| Self {
            system: list("system"),
            users: list("users"),
            summary,
            tail: list("tail"),
        })
    }
}

fn oa_item_type(it: &Value) -> &str {
    it.get("type").and_then(Value::as_str).unwrap_or("message")
}

/// A system or developer message item (kept verbatim, rendered first).
fn oa_system_message(it: &Value) -> bool {
    oa_item_type(it) == "message"
        && matches!(
            it.get("role").and_then(Value::as_str),
            Some("system" | "developer")
        )
}

/// One response's output as it comes back: reasoning, the assistant
/// message, its function calls.
fn oa_assistant_output(it: &Value) -> bool {
    match oa_item_type(it) {
        "reasoning" | "function_call" => true,
        "message" => it.get("role").and_then(Value::as_str) == Some("assistant"),
        _ => false,
    }
}

/// Take the state of the last compaction item when it is a v2 state: it
/// and every item before it leave `items` (the state stands for them -
/// whether the client pruned them or not). None, items untouched, when the
/// last compaction item is a legacy or failed one (or there is none):
/// [`oa_resend_rewrite`] handles those.
pub fn oa_take_state(items: &mut Vec<Value>) -> Option<OaState> {
    let k = items
        .iter()
        .rposition(|it| oa_item_type(it) == "compaction")?;
    let st = items[k]
        .get("encrypted_content")
        .and_then(Value::as_str)
        .and_then(OaState::decode)?;
    items.drain(..=k);
    Some(st)
}

/// A message item's content as Responses parts.
fn oa_parts(it: &Value) -> Vec<Value> {
    match it.get("content") {
        Some(Value::String(s)) => vec![json!({"type": "input_text", "text": s})],
        Some(Value::Array(parts)) => parts.clone(),
        _ => Vec::new(),
    }
}

/// The compacted window's one user message: the user's messages verbatim,
/// the framed summary, then any pending user items. One message, because
/// doubled user turns break strict-alternation templates (gemma's); the
/// separators are parts of their own because the Responses converter joins
/// text parts with nothing between them.
fn oa_head(users: &[Value], summary: &str, pending: &[&Value]) -> Value {
    let sep = || json!({"type": "input_text", "text": "\n\n"});
    let text = |t: String| json!({"type": "input_text", "text": t});
    let mut parts: Vec<Value> = Vec::new();
    for (i, u) in users.iter().enumerate() {
        if i == 0 {
            parts.push(text(OA_EARLIER_FRAME.into()));
        } else {
            parts.push(sep());
        }
        parts.extend(oa_parts(u));
    }
    if !parts.is_empty() {
        parts.push(sep());
    }
    parts.push(text(format!("{OA_COMPACTION_FRAME}{summary}")));
    for (i, p) in pending.iter().enumerate() {
        parts.push(sep());
        if i == 0 {
            parts.push(text(OA_CURRENT_FRAME.into()));
        }
        parts.extend(oa_parts(p));
    }
    json!({"type": "message", "role": "user", "content": parts})
}

/// The window a state renders, with the items that came after its item:
/// system items, the head (users + summary, absorbing the user items that
/// lead what follows), then the rest raw. The compacting request renders
/// `oa_render(state, [])` and a resend `oa_render(state, after)`; `after`
/// starts with that response's own output, never a user item, so the two
/// agree token for token up to the reply.
pub fn oa_render(st: &OaState, after: &[Value]) -> Vec<Value> {
    let rest: Vec<&Value> = st
        .tail
        .iter()
        .chain(after)
        .filter(|it| oa_item_type(it) != "compaction")
        .collect();
    let lead = rest.iter().take_while(|it| oa_user_message(it)).count();
    let mut out = st.system.clone();
    out.push(oa_head(&st.users, &st.summary, &rest[..lead]));
    out.extend(rest[lead..].iter().map(|it| (*it).clone()));
    out
}

/// Where the raw tail starts in `rest`: a pending user turn, else the
/// latest tool round (its outputs and the assistant output that called
/// them). A tail past `max_chars` is not worth keeping raw - compaction
/// exists to shed exactly that - so everything is summarized.
fn oa_tail_cut(rest: &[Value], max_chars: usize) -> usize {
    let n = rest.len();
    let mut i = n;
    while i > 0 && oa_user_message(&rest[i - 1]) {
        i -= 1;
    }
    if i == n {
        while i > 0 && oa_item_type(&rest[i - 1]) == "function_call_output" {
            i -= 1;
        }
        while i > 0 && oa_assistant_output(&rest[i - 1]) {
            i -= 1;
        }
    }
    let chars: usize = rest[i..].iter().map(|it| it.to_string().len()).sum();
    if chars > max_chars { n } else { i }
}

/// What an in-create compaction keeps raw, in serialized chars (~3.5-4 a
/// token). Archive compactions (trigger, /compact) keep every user message
/// and no tail.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OaKeep {
    /// the latest round (or pending user turn)
    pub tail_chars: usize,
    /// user messages kept verbatim, newest first; one that does not fit is
    /// summarized like everything else (a pasted document must not pin the
    /// window above the threshold forever)
    pub user_chars: usize,
}

impl OaKeep {
    /// Half the threshold in chars for the tail (~15-25% of it in tokens:
    /// code runs ~3-4 chars a token, dense records ~2), a quarter for the
    /// user messages: the compacted window lands well under the threshold,
    /// so the next compaction is many rounds away, and the round re-prefilled
    /// behind the new head stays small. A typical coding round (a few files
    /// read) stays raw; a bulk read of many files is summarized.
    pub fn for_threshold(threshold: u64) -> Self {
        let t = usize::try_from(threshold).unwrap_or(usize::MAX);
        Self {
            tail_chars: t / 2,
            user_chars: t / 4,
        }
    }
}

/// Keep user messages newest-first within `budget` chars (skipping any that
/// do not fit), in their original order. Returns them and how many were
/// left to the summary.
fn oa_keep_users(users: Vec<Value>, budget: usize) -> (Vec<Value>, usize) {
    let mut keep = vec![false; users.len()];
    let mut used = 0usize;
    for (i, u) in users.iter().enumerate().rev() {
        let c = u.to_string().len();
        if used + c <= budget {
            used += c;
            keep[i] = true;
        }
    }
    let dropped = keep.iter().filter(|k| !**k).count();
    let kept = users
        .into_iter()
        .zip(keep)
        .filter_map(|(u, k)| k.then_some(u))
        .collect();
    (kept, dropped)
}

/// A planned compaction: the state to emit once the summary is in, the
/// window items the summarization pass reads, and the live window (what
/// runs when the summary fails).
#[derive(Debug, Clone)]
pub struct OaPlan {
    pub state: OaState,
    pub span: Vec<Value>,
    pub window: Vec<Value>,
}

/// Plan compacting the window `prev` (the last state, if any) + `after`
/// (the items after it). `keep`: what in-create compaction keeps raw; None
/// archives (compaction_trigger, /compact: every user message verbatim, no
/// tail). None when there is nothing worth summarizing - an empty window,
/// or (in-create) a span of only kept messages, or one too small a share of
/// the window to bring it down (re-compacting every request for a few
/// hundred tokens is all cost).
pub fn oa_plan(prev: Option<&OaState>, after: &[Value], keep: Option<OaKeep>) -> Option<OaPlan> {
    let window = match prev {
        Some(st) => oa_render(st, after),
        None => after.to_vec(),
    };
    let rest: Vec<Value> = match prev {
        Some(st) => st.tail.iter().chain(after).cloned().collect(),
        None => after.to_vec(),
    };
    if window.is_empty() {
        return None;
    }
    let cut = keep.map_or(rest.len(), |k| oa_tail_cut(&rest, k.tail_chars));
    let (span_rest, tail) = rest.split_at(cut);
    let mut system = prev.map(|st| st.system.clone()).unwrap_or_default();
    let mut users = prev.map(|st| st.users.clone()).unwrap_or_default();
    // what the summary replaces: everything in the span but the kept
    // messages, plus the summary it supersedes
    let mut summarized = prev.map_or(0, |st| st.summary.len());
    for it in span_rest {
        if oa_system_message(it) {
            system.push(it.clone());
        } else if oa_user_message(it) {
            users.push(it.clone());
        } else {
            summarized += it.to_string().len();
        }
    }
    if let Some(k) = keep {
        let chars = |v: &[Value]| v.iter().map(|it| it.to_string().len()).sum::<usize>();
        let before = chars(&users);
        let (kept, dropped) = oa_keep_users(users, k.user_chars);
        summarized += before - chars(&kept);
        users = kept;
        let only_kept = dropped == 0
            && span_rest
                .iter()
                .all(|it| oa_user_message(it) || oa_system_message(it));
        if only_kept || summarized < chars(&window) / 4 {
            return None;
        }
    }
    // the tail is a suffix of the window (a cut inside the head's absorbed
    // user items leaves nothing to summarize, caught above)
    let span = window[..window.len() - tail.len()].to_vec();
    // Mid-turn (the tail is a tool round, not a pending user turn), the
    // request being worked on sits in the span: it moves to the front of the
    // tail, so the render places it after the summary as the current request
    // instead of in the list of earlier ones above it.
    // (Archives keep every user message in `users`: /compact echoes them.)
    let mut tail = tail.to_vec();
    if keep.is_some()
        && !tail.first().is_some_and(oa_user_message)
        && let Some(cur) = span_rest.iter().rev().find(|it| oa_user_message(it))
        && let Some(k) = users.iter().rposition(|u| u == cur)
    {
        tail.insert(0, users.remove(k));
    }
    Some(OaPlan {
        state: OaState {
            system,
            users,
            summary: String::new(),
            tail,
        },
        span,
        window,
    })
}

// -- Summarization in parts (a span past the summarizer's window) -----------
//
// One parallel round can outgrow the window on its own (live: an agent read
// ~14 files of ~12K tokens in one batch and sent a 294K-token request to a
// 262K model): the span the summary must cover then cannot be shown to the
// summarizer at all, the compaction failed, and the request was a 400. The
// pass now rolls over the span in parts (the refine pattern agent
// frameworks use for histories larger than the summarizer): part 1 is the
// span's own prefix - the live conversation's, so its prefill still rides
// the radix cache - and every later part opens with the user's messages
// verbatim plus the summary so far, which it updates. Cuts fall between
// items and never show a tool result without its call: a part that opens
// with results re-carries the calls that made them. An item that alone
// outgrows a part is cut down to fit, its middle elided and marked. (A
// map-then-merge or prefix-aware parallel layout spreads work across slots,
// but needs the history to fit the window - exactly what failed here.)

/// Instructions for the first part of a multi-part summarization.
pub const OA_PART_FIRST: &str = "\
The conversation above is about to be compacted to save context, and it is \
too long to summarize in one pass: what you see is its FIRST part, and the \
later parts will be shown to you next to update your summary. Write a \
structured, self-contained summary of this part: the user's goals and \
constraints, what has been done so far and why, the key facts and results \
worth carrying forward (names, values, findings, decisions - exactly as they \
appeared), errors already resolved, and what was in progress where this \
part ends. Reply with the summary only.";

/// Instructions for a later part (`last`: the final one).
pub fn oa_part_next(last: bool, keeps_tail: bool) -> String {
    let end = match (last, keeps_tail) {
        (false, _) => "More parts follow after this one.",
        (true, true) => {
            "This is the LAST part: your reply replaces the whole conversation \
             above except the user's own messages and the most recent tool round, \
             which stay as they are - include precisely what remains to be done next."
        }
        (true, false) => {
            "This is the LAST part: your reply replaces the whole conversation \
             above except the user's own messages - include what remains open."
        }
    };
    format!(
        "The conversation is being compacted in parts. The first message above \
         holds the user's own messages and the summary of everything before this \
         part; the messages after it are the NEXT part. Update the summary so it \
         covers everything so far, in the same structure: keep what still matters \
         from the summary, add the key facts and results from this part (names, \
         values, findings, decisions - exactly as they appeared), and note what \
         was in progress where this part ends. {end} Reply with the updated \
         summary only."
    )
}

/// The message a later part opens with: the user's messages verbatim and
/// the summary so far, absorbing any user items that lead the part (one
/// user turn - see `oa_head`).
pub fn oa_part_context(users: &[Value], summary: &str, leading: &[Value]) -> Value {
    let sep = || json!({"type": "input_text", "text": "\n\n"});
    let text = |t: String| json!({"type": "input_text", "text": t});
    let mut parts = vec![text(
        "This conversation is being compacted in parts.".into(),
    )];
    for (i, u) in users.iter().enumerate() {
        parts.push(sep());
        if i == 0 {
            parts.push(text(
                "The user's own messages so far (verbatim):\n\n".into(),
            ));
        }
        parts.extend(oa_parts(u));
    }
    parts.push(sep());
    parts.push(text(format!(
        "Summary of the conversation before the part that follows:\n\n{summary}"
    )));
    for p in leading {
        parts.push(sep());
        parts.extend(oa_parts(p));
    }
    json!({"type": "message", "role": "user", "content": parts})
}

fn oa_call_id(it: &Value) -> Option<&str> {
    it.get("call_id").and_then(Value::as_str)
}

/// `s` with its middle cut so `keep` (0..1) of its chars remain, marked.
fn oa_elide_str(s: &str, keep: f64) -> String {
    let n = s.chars().count();
    let k = ((n as f64) * keep.clamp(0.0, 1.0)) as usize;
    if k >= n {
        return s.to_owned();
    }
    let head: String = s.chars().take(k / 2).collect();
    let tail: String = s.chars().skip(n - (k - k / 2)).collect();
    format!(
        "{head}\n[... {} characters omitted here to fit the summarization pass ...]\n{tail}",
        n - k
    )
}

/// Elide the long strings inside a JSON value (tool-call arguments stay
/// valid JSON: they are parsed into objects at render time).
fn oa_elide_json(v: &mut Value, keep: f64, floor: usize) {
    match v {
        Value::String(s) if s.chars().count() > floor => *s = oa_elide_str(s, keep),
        Value::Array(a) => a.iter_mut().for_each(|x| oa_elide_json(x, keep, floor)),
        Value::Object(o) => o.values_mut().for_each(|x| oa_elide_json(x, keep, floor)),
        _ => {}
    }
}

/// An item cut down so about `keep` of its text remains: the long text
/// fields (tool output, message or reasoning text, call-argument strings)
/// lose their middle; pictures and short fields stay.
pub fn oa_elide(it: &Value, keep: f64) -> Value {
    let mut it = it.clone();
    let floor = 256;
    for key in ["output", "content", "summary"] {
        if let Some(v) = it.get_mut(key) {
            match v {
                Value::String(s) if s.chars().count() > floor => *s = oa_elide_str(s, keep),
                Value::Array(parts) => {
                    for p in parts.iter_mut() {
                        if let Some(Value::String(s)) = p.get_mut("text")
                            && s.chars().count() > floor
                        {
                            *s = oa_elide_str(s, keep);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    if let Some(Value::String(args)) = it.get("arguments")
        && let Ok(mut v) = serde_json::from_str::<Value>(args)
    {
        oa_elide_json(&mut v, keep, floor);
        it["arguments"] = json!(v.to_string());
    }
    it
}

/// Cut `span` into consecutive parts whose `cost` (tokens, estimated) fits
/// `budget`. Cuts fall between items; a part that opens with tool results
/// re-carries the calls that made them; an item that alone outgrows the
/// budget is elided to fit. One part when the span already fits.
pub fn oa_span_parts(
    span: &[Value],
    cost: &dyn Fn(&Value) -> usize,
    budget: usize,
) -> Vec<Vec<Value>> {
    let budget = budget.max(1);
    let mut parts: Vec<Vec<Value>> = Vec::new();
    let mut cur: Vec<Value> = Vec::new();
    let mut used = 0usize;
    let mut calls: std::collections::HashMap<String, Value> = Default::default();
    for it in span {
        let mut it = it.clone();
        let mut c = cost(&it);
        if c > budget {
            // room left for a re-carried call beside it
            it = oa_elide(&it, budget as f64 * 0.85 / c as f64);
            c = cost(&it);
        }
        if used + c > budget && !cur.is_empty() {
            parts.push(std::mem::take(&mut cur));
            used = 0;
        }
        match oa_item_type(&it) {
            "function_call" => {
                if let Some(id) = oa_call_id(&it) {
                    calls.insert(id.to_owned(), it.clone());
                }
            }
            "function_call_output" => {
                let id = oa_call_id(&it);
                let shown = cur
                    .iter()
                    .any(|x| oa_item_type(x) == "function_call" && oa_call_id(x) == id);
                if !shown && let Some(call) = id.and_then(|i| calls.get(i)) {
                    // before the run of results it joins: [calls.., results..]
                    let at = cur
                        .iter()
                        .rposition(|x| oa_item_type(x) != "function_call_output")
                        .map_or(0, |p| p + 1);
                    used += cost(call);
                    cur.insert(at, call.clone());
                }
            }
            _ => {}
        }
        used += c;
        cur.push(it);
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts
}

/// A tool_use block's location + identity, in conversation order.
struct ToolUse {
    msg: usize,
    block: usize,
    id: String,
    name: String,
}

/// Every tool_use block in assistant messages, oldest first.
fn tool_uses(messages: &[Value]) -> Vec<ToolUse> {
    let mut out = Vec::new();
    for (mi, m) in messages.iter().enumerate() {
        if m.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(blocks) = m.get("content").and_then(Value::as_array) else {
            continue;
        };
        for (bi, b) in blocks.iter().enumerate() {
            if b.get("type").and_then(Value::as_str) == Some("tool_use") {
                out.push(ToolUse {
                    msg: mi,
                    block: bi,
                    id: b.get("id").and_then(Value::as_str).unwrap_or("").to_owned(),
                    name: b
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned(),
                });
            }
        }
    }
    out
}

/// Replace the matching tool_result blocks' content with the placeholder.
/// Returns true when something actually changed (an already-cleared pair
/// changes nothing, which is what lets repeated requests converge).
fn clear_result(messages: &mut [Value], tool_use_id: &str) -> bool {
    let mut changed = false;
    for m in messages.iter_mut() {
        if m.get("role").and_then(Value::as_str) != Some("user") {
            continue;
        }
        let Some(blocks) = m.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for b in blocks.iter_mut() {
            if b.get("type").and_then(Value::as_str) == Some("tool_result")
                && b.get("tool_use_id").and_then(Value::as_str) == Some(tool_use_id)
            {
                let placeholder = json!([{"type": "text", "text": CLEARED_RESULT}]);
                if b.get("content") != Some(&placeholder) {
                    b["content"] = placeholder;
                    changed = true;
                }
            }
        }
    }
    changed
}

/// Apply the configured edits. `original_tokens` is the rendered token count
/// of the unedited prompt; `count` re-renders candidate message arrays so
/// triggers and reports use exact tokenizer counts, not estimates. Returns
/// the (possibly edited) messages and the report.
pub fn apply(
    cfg: &Config,
    messages: &[Value],
    original_tokens: usize,
    mut count: impl FnMut(&[Value]) -> Result<usize, String>,
) -> Result<(Vec<Value>, Applied), String> {
    let mut msgs: Vec<Value> = messages.to_vec();
    let mut edits_out: Vec<Value> = Vec::new();
    let mut tokens_now = original_tokens;
    let mut compact: Option<CompactFire> = None;

    for edit in &cfg.edits {
        match edit {
            Edit::Compact {
                trigger,
                instructions,
                pause,
            } => {
                // evaluated at its list position, so clears listed before it
                // get their chance to bring the prompt back under the trigger
                // (the cheap strategies first, the generation only as needed)
                if (tokens_now as u64) < *trigger {
                    continue;
                }
                // an empty span means the whole prompt is the pending turn -
                // nothing to summarize away; clear_tool_uses is the tool there
                if compact_tail_start(&msgs) == 0 {
                    continue;
                }
                compact = Some(CompactFire {
                    instructions: instructions.clone(),
                    pause: *pause,
                });
            }
            Edit::ClearThinking { keep_turns } => {
                let Some(keep) = keep_turns else { continue }; // keep all = no-op
                // a "thinking turn" is an assistant message holding >= 1
                // thinking block; keep the last `keep` of those intact
                let turn_idx: Vec<usize> = msgs
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| {
                        m.get("role").and_then(Value::as_str) == Some("assistant")
                            && m.get("content")
                                .and_then(Value::as_array)
                                .is_some_and(|blocks| {
                                    blocks.iter().any(|b| {
                                        matches!(
                                            b.get("type").and_then(Value::as_str),
                                            Some("thinking") | Some("redacted_thinking")
                                        )
                                    })
                                })
                    })
                    .map(|(i, _)| i)
                    .collect();
                if turn_idx.len() <= *keep as usize {
                    continue;
                }
                let clear_until = turn_idx[turn_idx.len() - *keep as usize];
                let mut cleared_turns = 0u64;
                for (mi, m) in msgs.iter_mut().enumerate() {
                    if mi >= clear_until {
                        break;
                    }
                    let Some(blocks) = m.get_mut("content").and_then(Value::as_array_mut) else {
                        continue;
                    };
                    let before = blocks.len();
                    blocks.retain(|b| {
                        !matches!(
                            b.get("type").and_then(Value::as_str),
                            Some("thinking") | Some("redacted_thinking")
                        )
                    });
                    if blocks.len() < before {
                        cleared_turns += 1;
                    }
                }
                if cleared_turns == 0 {
                    continue;
                }
                let after = count(&msgs)?;
                edits_out.push(json!({
                    "type": "clear_thinking_20251015",
                    "cleared_thinking_turns": cleared_turns,
                    "cleared_input_tokens": tokens_now.saturating_sub(after),
                }));
                tokens_now = after;
            }
            Edit::ClearToolUses {
                trigger,
                keep,
                clear_at_least,
                exclude_tools,
                clear_tool_inputs,
            } => {
                let uses = tool_uses(&msgs);
                let fired = match trigger {
                    Threshold::InputTokens(v) => tokens_now as u64 >= *v,
                    Threshold::ToolUses(v) => uses.len() as u64 >= *v,
                };
                if !fired {
                    continue;
                }
                // candidates: everything except the most recent `keep` pairs
                // and excluded tools - oldest first, which maximizes the
                // surviving cacheable prefix
                let keep_from = uses.len().saturating_sub(*keep as usize);
                let candidates: Vec<&ToolUse> = uses[..keep_from]
                    .iter()
                    .filter(|u| !exclude_tools.iter().any(|x| x == &u.name))
                    .collect();
                if candidates.is_empty() {
                    continue;
                }
                let mut edited = msgs.clone();
                let mut cleared = 0u64;
                for u in &candidates {
                    let mut changed = clear_result(&mut edited, &u.id);
                    if *clear_tool_inputs {
                        let b = &mut edited[u.msg]["content"][u.block];
                        if b.get("input") != Some(&json!({})) {
                            b["input"] = json!({});
                            changed = true;
                        }
                    }
                    if changed {
                        cleared += 1;
                    }
                }
                if cleared == 0 {
                    continue; // everything clearable was already placeholders
                }
                let after = count(&edited)?;
                let reclaimed = tokens_now.saturating_sub(after);
                // clear_at_least: an edit that cannot reclaim enough applies
                // nothing - a small clear still invalidates the whole prompt
                // cache after the first edited token, which is exactly the
                // waste this knob exists to refuse (pinned semantics; see
                // module docs)
                if let Some(min) = clear_at_least {
                    let enough = match min {
                        Threshold::InputTokens(v) => reclaimed as u64 >= *v,
                        Threshold::ToolUses(v) => cleared >= *v,
                    };
                    if !enough {
                        continue;
                    }
                }
                msgs = edited;
                edits_out.push(json!({
                    "type": "clear_tool_uses_20250919",
                    "cleared_tool_uses": cleared,
                    "cleared_input_tokens": reclaimed,
                }));
                tokens_now = after;
            }
        }
    }
    Ok((
        msgs,
        Applied {
            edits: edits_out,
            final_tokens: tokens_now,
            compact,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic fake "tokenizer": chars/4 over the serialized array -
    /// the core only compares counts, so any monotone measure exercises it.
    fn fake_count(msgs: &[Value]) -> Result<usize, String> {
        Ok(serde_json::to_string(msgs).unwrap().len() / 4)
    }

    fn tool_turn(id: &str, name: &str, result: &str) -> [Value; 2] {
        [
            json!({"role": "assistant", "content": [
                {"type": "tool_use", "id": id, "name": name, "input": {"q": "x"}}]}),
            json!({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": id, "content": result}]}),
        ]
    }

    fn cfg(v: Value) -> Config {
        parse(&v).expect("valid config")
    }

    #[test]
    fn trigger_unmet_is_a_noop() {
        let msgs: Vec<Value> = tool_turn("t1", "search", "big result").into();
        let c = cfg(json!({"edits": [{"type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": 1_000_000}}]}));
        let (out, applied) = apply(&c, &msgs, 500, fake_count).unwrap();
        assert!(applied.edits.is_empty());
        assert_eq!(out, msgs);
    }

    #[test]
    fn clears_oldest_keeps_recent_and_excluded() {
        let mut msgs: Vec<Value> = Vec::new();
        msgs.extend(tool_turn("t1", "lookup", &"x".repeat(400)));
        msgs.extend(tool_turn("t2", "web_search", &"y".repeat(400)));
        msgs.extend(tool_turn("t3", "lookup", &"z".repeat(400)));
        let c = cfg(json!({"edits": [{"type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 2},
            "keep": {"type": "tool_uses", "value": 1},
            "exclude_tools": ["web_search"]}]}));
        let orig = fake_count(&msgs).unwrap();
        let (out, applied) = apply(&c, &msgs, orig, fake_count).unwrap();
        assert_eq!(applied.edits.len(), 1);
        assert_eq!(
            applied.edits[0]["cleared_tool_uses"], 1,
            "t1 only: t3 kept, t2 excluded"
        );
        let r1 = out[1]["content"][0]["content"][0]["text"].as_str().unwrap();
        assert_eq!(r1, CLEARED_RESULT);
        assert!(
            out[3]["content"][0]["content"]
                .as_str()
                .unwrap()
                .contains('y'),
            "excluded intact"
        );
        assert!(
            out[5]["content"][0]["content"]
                .as_str()
                .unwrap()
                .contains('z'),
            "kept intact"
        );
        // inputs stay by default
        assert_eq!(out[0]["content"][0]["input"]["q"], "x");
        assert!(applied.edits[0]["cleared_input_tokens"].as_u64().unwrap() > 0);
    }

    #[test]
    fn clear_tool_inputs_also_empties_the_call() {
        let msgs: Vec<Value> = [
            tool_turn("t1", "a", &"x".repeat(200)).to_vec(),
            tool_turn("t2", "a", "tail").to_vec(),
        ]
        .concat();
        let c = cfg(json!({"edits": [{"type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 1},
            "keep": {"type": "tool_uses", "value": 1},
            "clear_tool_inputs": true}]}));
        let (out, applied) = apply(&c, &msgs, 9999, fake_count).unwrap();
        assert_eq!(applied.edits[0]["cleared_tool_uses"], 1);
        assert_eq!(out[0]["content"][0]["input"], json!({}));
    }

    #[test]
    fn clear_at_least_refuses_a_pointless_clear() {
        // one tiny clearable result - reclaiming it cannot meet the floor
        let msgs: Vec<Value> = [
            tool_turn("t1", "a", "tiny").to_vec(),
            tool_turn("t2", "a", "tail").to_vec(),
        ]
        .concat();
        let c = cfg(json!({"edits": [{"type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 1},
            "keep": {"type": "tool_uses", "value": 1},
            "clear_at_least": {"type": "input_tokens", "value": 100_000}}]}));
        let (out, applied) = apply(&c, &msgs, 9999, fake_count).unwrap();
        assert!(
            applied.edits.is_empty(),
            "small reclaim + cache invalidation = refused"
        );
        assert_eq!(out, msgs);
    }

    #[test]
    fn already_cleared_pairs_converge_to_noop() {
        let mut msgs: Vec<Value> = [
            tool_turn("t1", "a", &"x".repeat(200)).to_vec(),
            tool_turn("t2", "a", "tail").to_vec(),
        ]
        .concat();
        let c = cfg(json!({"edits": [{"type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 1},
            "keep": {"type": "tool_uses", "value": 1}}]}));
        let (once, applied) = apply(&c, &msgs, 9999, fake_count).unwrap();
        assert_eq!(applied.edits.len(), 1);
        msgs = once;
        let (again, applied2) = apply(&c, &msgs, 9999, fake_count).unwrap();
        assert!(
            applied2.edits.is_empty(),
            "re-clearing placeholders reports nothing"
        );
        assert_eq!(again, msgs);
    }

    #[test]
    fn clear_thinking_keeps_last_n_turns() {
        let think = |id: u32| {
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": format!("thought {id} {}", "t".repeat(100)), "signature": ""},
                {"type": "text", "text": format!("answer {id}")}]})
        };
        let user = json!({"role": "user", "content": "q"});
        let msgs = vec![
            user.clone(),
            think(1),
            user.clone(),
            think(2),
            user.clone(),
            think(3),
        ];
        let c = cfg(json!({"edits": [{"type": "clear_thinking_20251015",
            "keep": {"type": "thinking_turns", "value": 1}}]}));
        let orig = fake_count(&msgs).unwrap();
        let (out, applied) = apply(&c, &msgs, orig, fake_count).unwrap();
        assert_eq!(applied.edits.len(), 1);
        assert_eq!(applied.edits[0]["cleared_thinking_turns"], 2);
        // turns 1+2 lost their thinking blocks, answers intact, turn 3 whole
        assert_eq!(out[1]["content"].as_array().unwrap().len(), 1);
        assert_eq!(out[1]["content"][0]["type"], "text");
        assert_eq!(out[3]["content"].as_array().unwrap().len(), 1);
        assert_eq!(out[5]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn keep_all_thinking_is_a_noop_and_order_rule_enforced() {
        let c = cfg(json!({"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}));
        let msgs = vec![json!({"role": "user", "content": "q"})];
        let (_, applied) = apply(&c, &msgs, 10, fake_count).unwrap();
        assert!(applied.edits.is_empty());
        // clear_thinking anywhere but first is the documented error
        assert!(
            parse(&json!({"edits": [
                {"type": "clear_tool_uses_20250919"},
                {"type": "clear_thinking_20251015"}]}))
            .is_err()
        );
        // unknown strategies are refused
        assert!(parse(&json!({"edits": [{"type": "clear_everything_9000"}]})).is_err());
    }

    #[test]
    fn compact_parse_defaults_and_deviations() {
        // bare edit: default trigger, no instructions, no pause
        let c = cfg(json!({"edits": [{"type": "compact_20260112"}]}));
        assert!(matches!(
            &c.edits[0],
            Edit::Compact {
                trigger: 150_000,
                instructions: None,
                pause: false
            }
        ));
        // sub-50k triggers are ACCEPTED (documented deviation: a 32k-window
        // local server needs compaction well under Anthropic's floor)
        let c = cfg(json!({"edits": [{"type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 500},
            "instructions": "keep the numbers", "pause_after_compaction": true}]}));
        assert!(matches!(&c.edits[0],
            Edit::Compact { trigger: 500, instructions: Some(i), pause: true } if i == "keep the numbers"));
        // trigger is input_tokens only (SDK: BetaInputTokensTriggerParam)
        assert!(
            parse(&json!({"edits": [{"type": "compact_20260112",
                "trigger": {"type": "tool_uses", "value": 3}}]}))
            .is_err()
        );
        // one compact edit max
        assert!(
            parse(&json!({"edits": [
                {"type": "compact_20260112"}, {"type": "compact_20260112"}]}))
            .is_err()
        );
    }

    /// A grown conversation: two finished exchanges (one with a tool round),
    /// then the pending turn with a tool round under it.
    fn grown_conversation() -> Vec<Value> {
        let mut msgs = vec![
            json!({"role": "user", "content": "First question?"}),
            json!({"role": "assistant", "content": "First answer."}),
            json!({"role": "user", "content": "Second question?"}),
        ];
        msgs.extend(tool_turn("t1", "lookup", &"x".repeat(200)));
        msgs.push(json!({"role": "assistant", "content": "Second answer."}));
        msgs.push(
            json!({"role": "user", "content": [{"type": "text", "text": "Pending question?"}]}),
        );
        msgs.extend(tool_turn("t2", "lookup", "pending tool data"));
        msgs
    }

    #[test]
    fn compact_span_is_everything_before_the_pending_turn() {
        let msgs = grown_conversation();
        // the tail starts at the pending user question (index 6), not at the
        // t2 tool_result user message after it
        assert_eq!(compact_tail_start(&msgs), 6);
        // all-tool tails (no plain user message at all) have no split
        let only_tools: Vec<Value> = tool_turn("t9", "a", "r").into();
        assert_eq!(compact_tail_start(&only_tools), 0);
    }

    #[test]
    fn compact_fires_without_an_applied_edits_entry() {
        let msgs = grown_conversation();
        let c = cfg(json!({"edits": [{"type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 100}}]}));
        let (out, applied) = apply(&c, &msgs, 5000, fake_count).unwrap();
        assert!(applied.compact.is_some(), "trigger met, span non-empty");
        assert!(
            applied.edits.is_empty(),
            "SDK pin: compaction never in applied_edits"
        );
        assert_eq!(
            out, msgs,
            "the pure core does not touch messages for compact"
        );
        // unmet trigger
        let (_, applied) = apply(&c, &msgs, 50, fake_count).unwrap();
        assert!(applied.compact.is_none());
        // empty span: the whole prompt is the pending turn
        let pending_only = vec![json!({"role": "user", "content": "hi"})];
        let (_, applied) = apply(&c, &pending_only, 5000, fake_count).unwrap();
        assert!(applied.compact.is_none());
    }

    #[test]
    fn clears_listed_first_can_defuse_the_compact_trigger() {
        let mut msgs: Vec<Value> = Vec::new();
        msgs.extend(tool_turn("t1", "a", &"x".repeat(4000)));
        msgs.extend(tool_turn("t2", "a", "small"));
        msgs.push(json!({"role": "user", "content": "Pending?"}));
        let orig = fake_count(&msgs).unwrap();
        // clear reclaims ~1000 fake-tokens; trigger sits between post- and
        // pre-clear counts, so the compact edit sees the DEFUSED count
        let c = cfg(json!({"edits": [
            {"type": "clear_tool_uses_20250919",
             "trigger": {"type": "tool_uses", "value": 1},
             "keep": {"type": "tool_uses", "value": 1}},
            {"type": "compact_20260112",
             "trigger": {"type": "input_tokens", "value": (orig - 100) as u64}}]}));
        let (_, applied) = apply(&c, &msgs, orig, fake_count).unwrap();
        assert_eq!(applied.edits.len(), 1, "the clear applied");
        assert!(
            applied.compact.is_none(),
            "post-clear count is under the compact trigger"
        );
    }

    #[test]
    fn compacted_and_resend_render_the_identical_conversation() {
        // The cache-stability invariant: iteration 2's conversation must be a
        // prefix of what the next turn's resend rewrites to, or every
        // post-compaction turn is a cold prefill.
        let msgs = grown_conversation();
        let summary = "The user asked two questions; both were answered.";
        let in_request = compacted_messages(&msgs, summary);
        // tail = pending question + t2 round; summary framed into the front
        assert_eq!(in_request.len(), 3);
        let first_text = in_request[0]["content"][0]["text"].as_str().unwrap();
        assert!(first_text.starts_with(COMPACTION_FRAME));
        assert!(first_text.ends_with(summary));
        assert_eq!(in_request[0]["content"][1]["text"], "Pending question?");

        // the client appends our response (compaction block + answer) and a
        // new user turn, then resends the whole history
        let mut resend = msgs.clone();
        resend.push(json!({"role": "assistant", "content": [
            {"type": "compaction", "content": summary},
            {"type": "text", "text": "Final answer."}]}));
        resend.push(json!({"role": "user", "content": "Follow-up?"}));
        let rewritten = resend_rewrite(&resend).expect("compaction block present");
        assert_eq!(
            &rewritten[..in_request.len()],
            &in_request[..],
            "identical prefix"
        );
        assert_eq!(
            rewritten[in_request.len()],
            json!({"role": "assistant", "content": [{"type": "text", "text": "Final answer."}]}),
            "the compaction block is consumed, the answer survives"
        );
        assert_eq!(
            rewritten[in_request.len() + 1],
            json!({"role": "user", "content": "Follow-up?"})
        );
    }

    /// A grown Responses conversation: finished exchange with a tool round,
    /// then the pending user item with a tool round under it.
    fn oa_items() -> Vec<Value> {
        vec![
            json!({"role": "user", "content": "First question?"}),
            json!({"type": "message", "role": "assistant",
                   "content": [{"type": "output_text", "text": "First answer."}]}),
            json!({"type": "function_call", "call_id": "c1", "name": "lookup",
                   "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "c1", "output": "old data"}),
            json!({"type": "message", "role": "assistant",
                   "content": [{"type": "output_text", "text": "Second answer."}]}),
            json!({"role": "user", "content": [{"type": "input_text", "text": "Pending?"}]}),
            json!({"type": "function_call", "call_id": "c2", "name": "lookup",
                   "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "c2", "output": "fresh data"}),
        ]
    }

    #[test]
    fn oa_parse_validates_and_defaults() {
        assert_eq!(oa_parse(&[]).unwrap(), None);
        assert_eq!(
            oa_parse(&[json!({"type": "compaction"})]).unwrap(),
            Some(OA_DEFAULT_THRESHOLD)
        );
        assert_eq!(
            oa_parse(&[json!({"type": "compaction", "compact_threshold": null})]).unwrap(),
            Some(OA_DEFAULT_THRESHOLD)
        );
        // sub-50k accepted, same deviation as the Anthropic dialect
        assert_eq!(
            oa_parse(&[json!({"type": "compaction", "compact_threshold": 800})]).unwrap(),
            Some(800)
        );
        assert!(oa_parse(&[json!({"type": "compaction", "compact_threshold": 0})]).is_err());
        assert!(oa_parse(&[json!({"type": "truncation"})]).is_err());
        assert!(oa_parse(&[json!({"type": "compaction"}), json!({"type": "compaction"})]).is_err());
    }

    #[test]
    fn oa_tail_is_the_pending_user_item() {
        let items = oa_items();
        // the pending user item at 5, not the function_call_output after it
        assert_eq!(oa_tail_start(&items), 5);
        // no user item at all: no split
        assert_eq!(oa_tail_start(&items[2..4]), 0);
    }

    #[test]
    fn oa_trigger_must_be_final() {
        let mut items = oa_items();
        assert!(!oa_take_trigger(&mut items).unwrap());
        items.push(json!({"type": "compaction_trigger"}));
        assert!(oa_take_trigger(&mut items).unwrap());
        assert_eq!(items.len(), oa_items().len(), "trigger consumed");
        // anywhere else is the documented error
        let mut bad = oa_items();
        bad.insert(0, json!({"type": "compaction_trigger"}));
        assert!(oa_take_trigger(&mut bad).is_err());
    }

    #[test]
    fn oa_compacted_and_resend_render_the_identical_items() {
        // the same cache-stability invariant as the Anthropic pair, at item
        // level: iteration 2's list must be a prefix of the resend rewrite
        let items = oa_items();
        let summary = "One exchange happened; a lookup returned old data.";
        let in_request = oa_compacted_items(&items, summary);
        assert_eq!(in_request.len(), 3, "pending user + c2 round");
        let first = in_request[0]["content"][0]["text"].as_str().unwrap();
        assert!(first.starts_with(COMPACTION_FRAME) && first.contains(summary));
        assert_eq!(in_request[0]["content"][1]["text"], "Pending?");

        // client appends our output (compaction item first, then the answer)
        // and a new turn, then resends the whole item list
        let mut resend = items.clone();
        resend.push(json!({"id": "cp_1", "type": "compaction", "encrypted_content": summary}));
        resend.push(json!({"type": "message", "role": "assistant",
                           "content": [{"type": "output_text", "text": "Answer."}]}));
        resend.push(json!({"role": "user", "content": "Follow-up?"}));
        let rewritten = oa_resend_rewrite(&resend).expect("compaction item present");
        assert_eq!(
            &rewritten[..in_request.len()],
            &in_request[..],
            "identical prefix"
        );
        assert_eq!(
            rewritten.len(),
            in_request.len() + 2,
            "answer + follow-up survive"
        );

        // string-content pending turn wraps into parts without losing text
        let short = vec![
            json!({"role": "user", "content": "Old."}),
            json!({"type": "message", "role": "assistant",
                   "content": [{"type": "output_text", "text": "A."}]}),
            json!({"role": "user", "content": "Now?"}),
        ];
        let c = oa_compacted_items(&short, "s");
        assert_eq!(c[0]["content"][1]["text"], "Now?");
    }

    #[test]
    fn oa_failed_compaction_item_is_a_noop_on_resend() {
        let mut items = oa_items();
        items.push(json!({"type": "compaction", "encrypted_content": ""}));
        items.push(json!({"role": "user", "content": "Continue."}));
        let rewritten = oa_resend_rewrite(&items).expect("item present");
        assert_eq!(rewritten.len(), items.len() - 1, "only the item stripped");
        assert_eq!(rewritten[0], items[0]);
        // no compaction item: nothing to rewrite
        assert!(oa_resend_rewrite(&oa_items()).is_none());
    }

    #[test]
    fn failed_compaction_block_is_a_noop_on_resend() {
        // SDK pin: null content = failed compaction, round-trips as a no-op
        let mut msgs = grown_conversation();
        msgs.push(json!({"role": "assistant", "content": [
            {"type": "compaction", "content": null}]}));
        msgs.push(json!({"role": "user", "content": "Continue."}));
        let rewritten = resend_rewrite(&msgs).expect("block present");
        // nothing dropped, the block (and its emptied message) stripped
        assert_eq!(rewritten.len(), msgs.len() - 1);
        assert_eq!(rewritten[0], msgs[0]);
        assert_eq!(rewritten.last().unwrap(), &msgs[msgs.len() - 1]);
        // no compaction block anywhere: nothing to rewrite
        assert!(resend_rewrite(&grown_conversation()).is_none());
    }

    // -- v2 state --

    fn keep(tail_chars: usize) -> OaKeep {
        OaKeep {
            tail_chars,
            user_chars: 10_000,
        }
    }

    /// One user request and the agent loop under it: two tool rounds.
    fn oa_agent_loop() -> Vec<Value> {
        vec![
            json!({"role": "user", "content": "Fix the failing tests."}),
            json!({"type": "reasoning", "id": "rs_1", "summary": [],
                   "content": [{"type": "reasoning_text", "text": "run them first"}]}),
            json!({"type": "function_call", "call_id": "c1", "name": "run_shell",
                   "arguments": "{\"command\":\"make test\"}"}),
            json!({"type": "function_call_output", "call_id": "c1", "output": "3 failures"}),
            json!({"type": "reasoning", "id": "rs_2", "summary": [],
                   "content": [{"type": "reasoning_text", "text": "read the module"}]}),
            json!({"type": "function_call", "call_id": "c2", "name": "read_file",
                   "arguments": "{\"path\":\"base.py\"}"}),
            json!({"type": "function_call_output", "call_id": "c2", "output": "def f(): ..."}),
        ]
    }

    fn head_text(head: &Value) -> String {
        head["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect()
    }

    /// The case server-side compaction exists for: ONE user request whose
    /// tool rounds fill the window. It compacts inside that turn - the
    /// request kept verbatim after the summary, the latest round raw - and a
    /// resend renders the same window whether the client pruned everything
    /// before the item (as OpenAI documents it may) or sent it all back.
    #[test]
    fn oa_v2_compacts_inside_a_single_agent_turn() {
        let items = oa_agent_loop();
        let plan = oa_plan(None, &items, Some(keep(10_000))).expect("something to summarize");
        assert!(plan.state.users.is_empty(), "no earlier requests");
        let mut tail = vec![items[0].clone()];
        tail.extend_from_slice(&items[4..]);
        assert_eq!(
            plan.state.tail, tail,
            "the current request + the latest round"
        );
        assert_eq!(
            plan.span,
            items[..4].to_vec(),
            "the summary reads up to the round"
        );
        assert_eq!(plan.window, items);

        let st = OaState {
            summary: "Ran the tests: 3 failures.".into(),
            ..plan.state
        };
        let in_request = oa_render(&st, &[]);
        assert_eq!(in_request.len(), 4, "head + the raw round");
        assert_eq!(
            head_text(&in_request[0]),
            format!(
                "{OA_COMPACTION_FRAME}Ran the tests: 3 failures.\n\n{OA_CURRENT_FRAME}Fix the failing tests."
            )
        );
        assert_eq!(&in_request[1..], &items[4..]);

        let item = json!({"id": "cp_1", "type": "compaction", "encrypted_content": st.encode()});
        let next = [
            json!({"type": "function_call", "call_id": "c3", "name": "edit_file",
                   "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "c3", "output": "edited"}),
        ];
        let mut full = items.clone();
        full.push(item.clone());
        full.extend(next.iter().cloned());
        let mut pruned = vec![item];
        pruned.extend(next.iter().cloned());
        let a = oa_resend_rewrite(&full).expect("item present");
        assert_eq!(
            a,
            oa_resend_rewrite(&pruned).expect("item present"),
            "pruned or not"
        );
        assert_eq!(
            &a[..in_request.len()],
            &in_request[..],
            "the compacting render is a prefix"
        );
        assert_eq!(&a[in_request.len()..], &next[..]);

        // the request entry takes the state out and renders it the same way
        let mut taken = full.clone();
        let got = oa_take_state(&mut taken).expect("v2 state");
        assert_eq!(got, st);
        assert_eq!(taken, next.to_vec());
        assert_eq!(oa_render(&got, &taken), a);
    }

    /// Mid-turn with several requests in the window: the earlier ones are
    /// listed above the summary, the one being worked on comes after it -
    /// with all of them in one list a model answered the FIRST request after
    /// a compaction (live, 2026-10-12). Archives keep every user message
    /// where /compact echoes them.
    #[test]
    fn oa_v2_the_current_request_follows_the_summary_mid_turn() {
        let mut items = oa_items();
        items.extend(oa_agent_loop()[4..].iter().cloned());
        let plan = oa_plan(None, &items, Some(keep(10_000))).expect("plan");
        assert_eq!(
            plan.state.users,
            vec![items[0].clone()],
            "the earlier request"
        );
        assert_eq!(
            plan.state.tail[0], items[5],
            "the pending request leads the tail"
        );
        let st = OaState {
            summary: "S".into(),
            ..plan.state
        };
        let w = oa_render(&st, &[]);
        assert_eq!(
            head_text(&w[0]),
            format!(
                "{OA_EARLIER_FRAME}First question?\n\n{OA_COMPACTION_FRAME}S\n\n{OA_CURRENT_FRAME}Pending?"
            )
        );
        assert_eq!(&w[1..], &items[items.len() - 3..]);
        let archive = oa_plan(None, &items, None).unwrap();
        assert_eq!(archive.state.users.len(), 2);
        assert!(archive.state.tail.is_empty());
    }

    /// A pending user turn is the raw tail: it renders after the summary,
    /// inside the one head message (no doubled user turn).
    #[test]
    fn oa_v2_a_pending_user_turn_follows_the_summary() {
        let mut items = oa_items();
        items.push(json!({"type": "message", "role": "assistant",
                          "content": [{"type": "output_text", "text": "Done."}]}));
        items.push(json!({"role": "user", "content": "Next?"}));
        let plan = oa_plan(None, &items, Some(keep(10_000))).expect("plan");
        assert_eq!(plan.state.users.len(), 2, "both earlier user messages kept");
        assert_eq!(plan.state.tail, vec![items[items.len() - 1].clone()]);
        let st = OaState {
            summary: "S".into(),
            ..plan.state
        };
        let w = oa_render(&st, &[]);
        assert_eq!(w.len(), 1, "one head message");
        assert_eq!(
            head_text(&w[0]),
            format!(
                "{OA_EARLIER_FRAME}First question?\n\nPending?\n\n{OA_COMPACTION_FRAME}S\n\n{OA_CURRENT_FRAME}Next?"
            )
        );
    }

    /// A second compaction reads the first summary (inside the head) and
    /// carries the kept messages forward.
    #[test]
    fn oa_v2_recompaction_carries_the_kept_messages_forward() {
        let items = oa_agent_loop();
        let plan = oa_plan(None, &items, Some(keep(10_000))).unwrap();
        let st1 = OaState {
            summary: "S1".into(),
            ..plan.state
        };
        let after: Vec<Value> = oa_agent_loop()[1..]
            .iter()
            .map(|it| {
                let mut it = it.clone();
                if let Some(c) = it.get("call_id").and_then(Value::as_str) {
                    it["call_id"] = json!(format!("{c}b"));
                }
                it
            })
            .collect();
        let plan2 = oa_plan(Some(&st1), &after, Some(keep(10_000))).expect("plan");
        assert!(
            plan2.state.users.is_empty(),
            "the request is still the current one"
        );
        let mut tail = vec![items[0].clone()];
        tail.extend_from_slice(&after[3..]);
        assert_eq!(plan2.state.tail, tail);
        assert!(
            head_text(&plan2.span[0]).contains("S1"),
            "the old summary is read"
        );
        assert_eq!(plan2.window, oa_render(&st1, &after));
        assert_eq!(plan2.span, plan2.window[..plan2.window.len() - 3].to_vec());
    }

    /// A raw round past the budget defeats the point: summarize it too.
    /// Nothing but kept messages before the tail: nothing to compact
    /// (unless archiving).
    #[test]
    fn oa_v2_big_tails_are_summarized_and_empty_spans_skip() {
        let items = oa_agent_loop();
        let plan = oa_plan(None, &items, Some(keep(10))).expect("plan");
        assert_eq!(
            plan.state.tail,
            vec![items[0].clone()],
            "no raw round, the request after"
        );
        assert_eq!(plan.span, items);

        let one_round = &items[..4];
        assert!(oa_plan(None, one_round, Some(keep(10_000))).is_none());
        let archive = oa_plan(None, one_round, None).expect("archive plan");
        assert!(archive.state.tail.is_empty());
        assert_eq!(archive.span, one_round.to_vec());
        assert!(oa_plan(None, &[], None).is_none());
    }

    /// System/developer items are kept verbatim and render first.
    #[test]
    fn oa_v2_keeps_developer_messages_first() {
        let mut items = oa_agent_loop();
        items.insert(0, json!({"role": "developer", "content": "Be brief."}));
        let plan = oa_plan(None, &items, Some(keep(10_000))).unwrap();
        assert_eq!(plan.state.system, vec![items[0].clone()]);
        let st = OaState {
            summary: "S".into(),
            ..plan.state
        };
        assert_eq!(oa_render(&st, &[])[0], items[0]);
    }

    /// The state round-trips; a plaintext (legacy) summary or a failed
    /// empty item is not a state, and keeps the old rewrite.
    #[test]
    fn oa_v2_state_round_trips_and_legacy_items_stay_legacy() {
        let st = OaState {
            system: vec![],
            users: vec![json!({"role": "user", "content": "q"})],
            summary: "S".into(),
            tail: vec![json!({"type": "function_call_output", "call_id": "c", "output": "o"})],
        };
        assert_eq!(OaState::decode(&st.encode()), Some(st));
        assert!(OaState::decode("A plain summary.").is_none());
        assert!(OaState::decode("").is_none());
        assert!(OaState::decode(r#"{"paddock_compaction":2,"summary":"  "}"#).is_none());
        let mut legacy = oa_items();
        legacy.push(json!({"type": "compaction", "encrypted_content": "plain"}));
        assert!(oa_take_state(&mut legacy).is_none());
        assert_eq!(legacy.len(), oa_items().len() + 1, "untouched");
    }

    /// A pasted document in a user message does not pin the window above
    /// the threshold: past the user budget it is summarized like the rest,
    /// while short user messages stay verbatim.
    #[test]
    fn oa_v2_a_pasted_document_is_summarized_not_pinned() {
        let doc = format!("Data log: {}", "row=7 ".repeat(1500));
        let items = vec![
            json!({"role": "user", "content": "Remember: the vault code is 4711."}),
            json!({"type": "message", "role": "assistant",
                   "content": [{"type": "output_text", "text": "Noted."}]}),
            json!({"role": "user", "content": doc}),
            json!({"type": "message", "role": "assistant",
                   "content": [{"type": "output_text", "text": "Noted the log."}]}),
            json!({"role": "user", "content": "What is the vault code?"}),
        ];
        let plan = oa_plan(None, &items, Some(OaKeep::for_threshold(800))).expect("plan");
        assert_eq!(
            plan.state.users,
            vec![items[0].clone()],
            "the document is summarized"
        );
        assert_eq!(plan.state.tail, vec![items[4].clone()]);
        assert_eq!(plan.span, items[..4].to_vec(), "the summary still reads it");
    }

    /// Nothing worth it: when the span a summary would replace is a small
    /// share of the window (a huge raw round kept), compaction is skipped
    /// rather than re-run every request for a few hundred tokens.
    #[test]
    fn oa_v2_skips_a_compaction_that_cannot_shrink_the_window() {
        let mut items = oa_agent_loop();
        items[6]["output"] = json!("x".repeat(20_000));
        let k = OaKeep {
            tail_chars: 50_000,
            user_chars: 10_000,
        };
        assert!(oa_plan(None, &items, Some(k)).is_none());
        // a round past its budget is summarized instead, which is worth it
        let k = OaKeep {
            tail_chars: 1_000,
            user_chars: 10_000,
        };
        let p = oa_plan(None, &items, Some(k)).expect("plan");
        assert_eq!(p.state.tail, vec![items[0].clone()]);
    }

    // -- summarization in parts --

    fn chars_cost(it: &Value) -> usize {
        it.to_string().len()
    }

    fn texts(it: &Value) -> String {
        it.to_string()
    }

    /// A span past the budget is cut between items, in order, and a part
    /// that opens with tool results re-carries the calls that made them - a
    /// result is never shown without its call.
    #[test]
    fn oa_parts_cut_between_items_and_recarry_calls() {
        let out = |id: &str| json!({"type": "function_call_output", "call_id": id, "output": "r".repeat(400)});
        let call = |id: &str| json!({"type": "function_call", "call_id": id, "name": "read_file", "arguments": "{}"});
        let span = vec![
            json!({"role": "user", "content": "Read all three."}),
            call("c1"),
            call("c2"),
            call("c3"),
            out("c1"),
            out("c2"),
            out("c3"),
        ];
        let one = oa_span_parts(&span, &chars_cost, 100_000);
        assert_eq!(
            one,
            vec![span.clone()],
            "a span that fits is one part, untouched"
        );

        // user + 3 calls + one result fit 800; each later result needs a part
        let parts = oa_span_parts(&span, &chars_cost, 800);
        assert_eq!(parts.len(), 3, "{parts:#?}");
        assert_eq!(
            parts[0],
            span[..5].to_vec(),
            "the prefix part: calls + the first result"
        );
        assert_eq!(parts[1], vec![call("c2"), out("c2")], "the call re-carried");
        assert_eq!(parts[2], vec![call("c3"), out("c3")]);
        // every result shows after its call, and none is lost or reordered
        for p in &parts {
            for (i, it) in p.iter().enumerate() {
                if it["type"] == "function_call_output" {
                    assert!(
                        p[..i]
                            .iter()
                            .any(|x| x["type"] == "function_call" && x["call_id"] == it["call_id"])
                    );
                }
            }
        }
        let results: Vec<&Value> = parts
            .iter()
            .flatten()
            .filter(|it| it["type"] == "function_call_output")
            .collect();
        assert_eq!(results, vec![&span[4], &span[5], &span[6]]);

        // two results of one batch in a later part: their calls lead it
        let parts = oa_span_parts(&span, &chars_cost, 1_000);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[1], vec![call("c2"), call("c3"), out("c2"), out("c3")]);
    }

    /// An item that alone outgrows a part is cut down to fit: its middle is
    /// elided and marked, head and tail kept.
    #[test]
    fn oa_an_item_past_the_budget_is_elided_to_fit() {
        let body = format!("HEAD{}TAIL", "x".repeat(10_000));
        let span = vec![json!({"type": "function_call_output", "call_id": "c1", "output": body})];
        let parts = oa_span_parts(&span, &chars_cost, 2_000);
        assert_eq!(parts.len(), 1);
        let it = &parts[0][0];
        assert!(chars_cost(it) <= 2_000, "fits: {}", chars_cost(it));
        let o = it["output"].as_str().unwrap();
        assert!(o.starts_with("HEAD") && o.ends_with("TAIL"));
        assert!(o.contains("characters omitted here to fit the summarization pass"));
    }

    /// Elided call arguments stay valid JSON (they are parsed into objects
    /// at render time); short fields are untouched.
    #[test]
    fn oa_elide_keeps_call_arguments_valid_json() {
        let args = json!({"path": "docs/notes.md", "content": "y".repeat(5_000)}).to_string();
        let it = json!({"type": "function_call", "call_id": "c1", "name": "write_file",
                        "arguments": args});
        let e = oa_elide(&it, 0.1);
        let a: Value = serde_json::from_str(e["arguments"].as_str().unwrap()).expect("JSON");
        assert_eq!(a["path"], "docs/notes.md");
        assert!(a["content"].as_str().unwrap().contains("omitted here"));
        assert!(texts(&e).len() < texts(&it).len() / 2);
        // reasoning and message text parts too
        let r = json!({"type": "reasoning", "id": "rs_1", "summary": [],
                       "content": [{"type": "reasoning_text", "text": "z".repeat(3_000)}]});
        assert!(
            oa_elide(&r, 0.2)["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("omitted")
        );
    }

    /// A later part opens with the user's messages verbatim and the summary
    /// so far, absorbing user items that lead the part (one user turn).
    #[test]
    fn oa_a_later_part_opens_with_users_and_the_summary_so_far() {
        let users = [
            json!({"role": "user", "content": "Audit the ledger."}),
            json!({"role": "user", "content": "Now the totals."}),
        ];
        let lead = [json!({"role": "user", "content": "And the routes."})];
        let c = oa_part_context(&users, "S so far", &lead);
        assert_eq!(c["role"], "user");
        let t: String = c["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect();
        let (a, b, x, y) = (
            t.find("Audit the ledger.").unwrap(),
            t.find("Now the totals.").unwrap(),
            t.find("S so far").unwrap(),
            t.find("And the routes.").unwrap(),
        );
        assert!(a < b && b < x && x < y, "{t}");
        assert!(oa_part_next(true, true).contains("LAST part"));
        assert!(oa_part_next(false, true).contains("More parts follow"));
    }
}
