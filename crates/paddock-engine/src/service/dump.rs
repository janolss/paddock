//! `PADDOCK_TOKEN_DUMP=<dir>` (development): each request's exact token
//! stream to a file - the prompt as the engine received it, the tokens it
//! generated and how it ended. A reply can then be replayed token for token: a
//! token-id prompt on `/v1/completions` scores any prefix of it, under any
//! runner configuration (KV dtype, attention mode), which is how a suspect
//! step's distribution is compared across numerics.
//!
//! The scheduler is untouched: the request's event channel is tapped at
//! submit by a task that forwards every event unchanged, and a client that
//! goes away still reaches the engine as a failed send.

use std::path::PathBuf;
use std::sync::OnceLock;

use super::{GenRequest, TokenEvent};

fn dump_dir() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_TOKEN_DUMP").map(PathBuf::from))
        .as_ref()
}

/// Tap `req`'s events when dumping is on (and a runtime is there to forward
/// them); a no-op otherwise.
pub(super) fn tap(req: &mut GenRequest) {
    let Some(dir) = dump_dir() else {
        return;
    };
    let Ok(rt) = tokio::runtime::Handle::try_current() else {
        return;
    };
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let out = std::mem::replace(&mut req.events, tx);
    let prompt = req.prompt.clone();
    let dir = dir.clone();
    let at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    rt.spawn(async move {
        let mut generated = Vec::new();
        let mut finish = "closed".to_owned();
        while let Some(ev) = rx.recv().await {
            let end = match &ev {
                TokenEvent::Token { id, .. } => {
                    generated.push(*id);
                    false
                }
                TokenEvent::Done(r, _) => {
                    finish = format!("{r:?}");
                    true
                }
                TokenEvent::Error(e) => {
                    finish = format!("error: {}", e.message);
                    true
                }
                _ => false,
            };
            if out.send(ev).is_err() {
                finish = "client gone".to_owned();
                break;
            }
            if end {
                break;
            }
        }
        let ids = |v: &[u32]| v.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let rec = format!(
            "{{\"submitted_unix_ms\":{at_ms},\"prompt_len\":{},\"generated_len\":{},\
             \"finish\":{},\"prompt\":[{}],\"generated\":[{}]}}",
            prompt.len(),
            generated.len(),
            json_str(&finish),
            ids(&prompt),
            ids(&generated)
        );
        let path = dir.join(format!("{at_ms}-{}.json", prompt.len()));
        if let Err(e) = std::fs::write(&path, rec) {
            tracing::warn!("token dump: {}: {e}", path.display());
        }
    });
}

/// `s` as a JSON string literal.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::json_str;

    #[test]
    fn finish_text_is_a_json_string() {
        assert_eq!(json_str("error: \"x\"\n"), r#""error: \"x\"\u000a""#);
        assert_eq!(json_str("a\\b"), r#""a\\b""#);
    }
}
