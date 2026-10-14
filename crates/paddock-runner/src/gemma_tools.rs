//! Gemma 4 tool calls: `<|tool_call>call:NAME{key:value,...}<tool_call|>`.
//!
//! The chat template writes a call's arguments in its own value syntax, not
//! JSON: keys are bare (`path:`), strings are wrapped in the `<|"|>` special
//! token with nothing escaped inside, objects and arrays nest with `{}` / `[]`,
//! and booleans and numbers are bare. The model writes calls the same way and
//! ends the turn on `<|tool_response>` (an eos in the checkpoint's
//! generation_config). transformers reads the arguments as lax JSON - pull the
//! `<|"|>` strings out, quote `\w+:` keys, `json.loads` - and this reader
//! accepts the same text, plus the template's own escaped-key form.
//!
//! It converts in generation order, so a growing call has a settled JSON
//! prefix at every token: a string value streams as it arrives (JSON-escaped,
//! holding back a tail that could still become a closing marker), while a key
//! or a bare scalar goes out once its delimiter arrives. The final parse runs
//! the same converter with nothing growing, so what streamed is always a
//! prefix of the final arguments.

use crate::parsers::{G_CLOSE, G_THOUGHT, Parsed, ToolCallRaw, holdback};
use crate::tool_stream::CallProgress;

pub(crate) const CALL: &str = "<|tool_call>";
pub(crate) const CALL_END: &str = "<tool_call|>";
/// What the model writes after its calls: the turn ends here (stop token).
pub(crate) const RESPONSE: &str = "<|tool_response>";
pub(crate) const STR: &str = "<|\"|>";
const NAME_PREFIX: &str = "call:";

/// What could still close an open string value.
const STOPS: &[&str] = &[STR, CALL_END];

#[derive(Clone, Copy, PartialEq)]
enum Ctx {
    Obj,
    Arr,
}

/// One block's text after `<|tool_call>` as `(name, arguments JSON, complete)`,
/// None until the name settles (the `{` after it) or when the block is not a
/// call. `growing`: more text may follow, so the arguments stop where the
/// text stops settling them; `complete` is false then. Final (`!growing`):
/// the whole arguments object, a truncated one closed best-effort, and only
/// when it is valid JSON.
pub(crate) fn call_json(block: &str, growing: bool) -> Option<(String, String, bool)> {
    let body = block.trim_start().strip_prefix(NAME_PREFIX)?;
    let brace = body.find('{')?;
    let name = body[..brace].trim();
    if name.is_empty() || name.contains(char::is_whitespace) {
        return None;
    }
    let (args, closed) = convert(&body[brace..], growing);
    if growing {
        return Some((name.to_owned(), args, false));
    }
    serde_json::from_str::<serde_json::Value>(&args).ok()?;
    Some((name.to_owned(), args, closed))
}

/// Gemma value syntax -> JSON text, from the opening `{`. Returns the JSON
/// (a settled prefix when `growing`) and whether the top-level object closed.
fn convert(s: &str, growing: bool) -> (String, bool) {
    let mut out = String::new();
    let mut stack: Vec<Ctx> = Vec::new();
    let mut key_next = false;
    // a key went out and its `:` has not
    let mut pending_key = false;
    let mut i = 0;
    let push_str = |out: &mut String, v: &str| {
        out.push_str(&serde_json::Value::String(v.to_owned()).to_string());
    };
    while i < s.len() {
        let rest = &s[i..];
        let c = rest.chars().next().expect("non-empty");
        if c.is_whitespace() {
            i += c.len_utf8();
            continue;
        }
        if growing && rest.len() < STR.len() && STR.starts_with(rest) {
            // a string marker still arriving
            return (out, false);
        }
        if let Some(after) = rest.strip_prefix(STR) {
            match after.find(STR) {
                Some(j) => {
                    push_str(&mut out, &after[..j]);
                    pending_key = key_next;
                    key_next = false;
                    i += STR.len() + j + STR.len();
                    continue;
                }
                None if growing => {
                    // an open string: its settled part streams, as a value
                    if !key_next {
                        let settled = &after[..after.len() - holdback(after, STOPS)];
                        let q = serde_json::Value::String(settled.to_owned()).to_string();
                        out.push_str(&q[..q.len() - 1]);
                    }
                    return (out, false);
                }
                None => {
                    // truncated mid-string: the rest of the text is the value
                    push_str(&mut out, after);
                    pending_key = key_next;
                    break;
                }
            }
        }
        match c {
            '{' | '[' => {
                out.push(c);
                stack.push(if c == '{' { Ctx::Obj } else { Ctx::Arr });
                key_next = c == '{';
                i += 1;
            }
            '}' | ']' => {
                // a trailing comma would make the JSON invalid
                if out.ends_with(',') {
                    out.pop();
                }
                out.push(c);
                stack.pop();
                key_next = false;
                i += 1;
                if stack.is_empty() {
                    return (out, true);
                }
            }
            ',' => {
                out.push(',');
                key_next = stack.last() == Some(&Ctx::Obj);
                i += 1;
            }
            ':' => {
                out.push(':');
                key_next = false;
                pending_key = false;
                i += 1;
            }
            _ => {
                // a bare key or scalar runs to the next delimiter
                let end = rest
                    .find(|ch: char| ",:{}[]".contains(ch) || ch.is_whitespace() || ch == '<')
                    .unwrap_or(rest.len());
                if end == rest.len() && growing {
                    return (out, false);
                }
                if end == 0 {
                    // a stray `<` that is not a string marker: treat as text
                    let n = c.len_utf8();
                    push_str(&mut out, &rest[..n]);
                    i += n;
                    continue;
                }
                let word = &rest[..end];
                if key_next {
                    push_str(&mut out, word);
                    pending_key = true;
                    key_next = false;
                } else {
                    out.push_str(&scalar(word));
                }
                i += end;
            }
        }
    }
    if growing {
        return (out, false);
    }
    // truncated (max_tokens mid-call): close what is open
    if out.ends_with(',') {
        out.pop();
    }
    if pending_key {
        out.push_str(":null");
    } else if out.ends_with(':') {
        out.push_str("null");
    }
    while let Some(ctx) = stack.pop() {
        out.push(if ctx == Ctx::Obj { '}' } else { ']' });
    }
    (out, false)
}

/// A bare scalar as JSON: a JSON number or literal passes through, the
/// template's Python spellings map to theirs, anything else is a string.
fn scalar(word: &str) -> String {
    match word {
        "true" | "True" => "true".into(),
        "false" | "False" => "false".into(),
        "null" | "None" | "none" => "null".into(),
        w if serde_json::from_str::<serde_json::Number>(w).is_ok() => w.into(),
        w => serde_json::Value::String(w.to_owned()).to_string(),
    }
}

/// Scan a region for call blocks: parsed calls land in `out` and the return
/// is the region's text with the blocks (and a trailing `<|tool_response>`)
/// removed. An unterminated final block still parses best-effort, but only
/// closed blocks count complete. A block that does not parse stays text.
pub(crate) fn scan_calls(region: &str, out: &mut Parsed) -> String {
    let mut kept = String::new();
    let mut cur = region;
    while let Some(s) = cur.find(CALL) {
        kept.push_str(&cur[..s]);
        let after = &cur[s + CALL.len()..];
        let (block, next, closed) = match after.find(CALL_END) {
            Some(e) => (&after[..e], &after[e + CALL_END.len()..], true),
            None => (after, "", false),
        };
        match call_json(block, false) {
            Some((name, arguments, _)) => {
                out.tool_calls.push(ToolCallRaw { name, arguments });
                if closed {
                    out.complete_calls = out.tool_calls.len();
                }
            }
            // a closed block that is no call stays text; an open one is held
            // back - it may still become a call, and streamed text cannot be
            // taken back
            None if closed => {
                kept.push_str(CALL);
                kept.push_str(block);
                kept.push_str(CALL_END);
            }
            None => {}
        }
        cur = next;
    }
    kept.push_str(cur);
    match kept.trim_end().strip_suffix(RESPONSE) {
        Some(k) => k.to_owned(),
        None => kept,
    }
}

/// Every call in a Gemma 4 turn so far, in the order the final parse returns
/// them: the thought channel's, then the answer's.
pub(crate) fn progress(text: &str, thinking_open: bool) -> Vec<CallProgress> {
    let mut out = Vec::new();
    let trimmed = text.trim_start();
    let thought = match trimmed.strip_prefix(G_THOUGHT) {
        Some(a) => Some(a),
        None if thinking_open => Some(trimmed),
        None => None,
    };
    match thought {
        Some(after) => match after.find(G_CLOSE) {
            Some(end) => {
                scan_progress(&after[..end], false, &mut out);
                scan_progress(&after[end + G_CLOSE.len()..], true, &mut out);
            }
            None => scan_progress(after, true, &mut out),
        },
        None => scan_progress(trimmed, true, &mut out),
    }
    out
}

/// The call blocks of one region. `growing`: the region's end is not final -
/// its last unclosed block may still be written.
fn scan_progress(region: &str, growing: bool, out: &mut Vec<CallProgress>) {
    let mut cur = region;
    while let Some(s) = cur.find(CALL) {
        let after = &cur[s + CALL.len()..];
        let (block, next, closed) = match after.find(CALL_END) {
            Some(e) => (&after[..e], &after[e + CALL_END.len()..], true),
            None => (after, "", false),
        };
        let grows = growing && !closed;
        if let Some((name, args, _)) = call_json(block, grows) {
            out.push(CallProgress {
                name,
                args,
                complete: !grows,
            });
        }
        cur = next;
    }
}

#[cfg(test)]
#[path = "gemma_tools_tests.rs"]
mod tests;
