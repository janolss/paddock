use serde_json::{Value, json};

use super::*;
use crate::parsers::{Dialect, parse};

/// The template the registry's gemma-4-26B-A4B-it Q8_0 GGUF embeds (what
/// serves), and the older Gemma 4 template the parser tests started from.
const TPLS: [(&str, &str); 2] = [
    (
        "26b-a4b gguf",
        include_str!("../tests/fixtures/gemma4_26b_a4b_chat_template.jinja"),
    ),
    (
        "older",
        include_str!("../tests/fixtures/gemma4_chat_template.jinja"),
    ),
];

fn args_of(text: &str) -> Value {
    let (_, a, _) = call_json(text, false).expect("a call");
    serde_json::from_str(&a).expect("JSON")
}

#[test]
fn calls_read_as_json_like_transformers_does() {
    let (name, a, done) = call_json(
        "call:read_file{path:<|\"|>/a b/c.py<|\"|>,limit:100}",
        false,
    )
    .unwrap();
    assert_eq!(name, "read_file");
    assert!(done);
    assert_eq!(
        serde_json::from_str::<Value>(&a).unwrap(),
        json!({"path": "/a b/c.py", "limit": 100})
    );
    // nesting, arrays, literals, raw quotes/newlines inside strings, and the
    // template's escaped-key form
    assert_eq!(
        args_of(
            "call:edit{edits:[{old:<|\"|>a\"b\n<|\"|>,new:<|\"|>c<|\"|>}],dry:false,n:-1.5e2,\
             meta:{<|\"|>k k<|\"|>:null}}"
        ),
        json!({"edits": [{"old": "a\"b\n", "new": "c"}], "dry": false, "n": -150.0,
               "meta": {"k k": null}})
    );
    // whitespace between tokens is not data
    assert_eq!(
        args_of("call:f{ a : 1 , b : [ true , None ] }"),
        json!({"a": 1, "b": [true, null]})
    );
    assert_eq!(args_of("call:noargs{}"), json!({}));
}

#[test]
fn a_truncated_call_closes_best_effort() {
    assert_eq!(
        args_of("call:Write{file_path:<|\"|>x.md<|\"|>,content:<|\"|>abc"),
        json!({"file_path": "x.md", "content": "abc"})
    );
    assert_eq!(args_of("call:f{a:1,b:"), json!({"a": 1, "b": null}));
    assert_eq!(args_of("call:f{a:1,b"), json!({"a": 1, "b": null}));
    assert_eq!(args_of("call:f{a:[1,2"), json!({"a": [1, 2]}));
    assert_eq!(
        args_of("call:f{a:1,<|\"|>half key"),
        json!({"a": 1, "half key": null})
    );
    // no name yet, or no call at all
    assert!(call_json("call:Re", false).is_none());
    assert!(call_json("something else", false).is_none());
}

/// Every cut of a call (at token boundaries: the specials are atomic, so
/// cut only between them and inside plain text) has a growing JSON that is
/// a prefix of the final arguments.
#[test]
fn growing_arguments_are_prefixes_of_the_final_ones() {
    let calls = [
        "call:Write{content:<|\"|>line one\nline \"two\" \\ end<|\"|>,file_path:<|\"|>/tmp/x<|\"|>}",
        "call:Edit{edits:[{a:1,b:<|\"|>q<|\"|>},{c:[true,false,null]}],replace_all:true}",
        "call:Bash{command:<|\"|>ls -la | grep \"x\" && echo done<|\"|>,timeout:120000}",
    ];
    for text in calls {
        let (_, fin, _) = call_json(text, false).unwrap();
        let cuts: Vec<usize> = text
            .char_indices()
            .map(|(i, _)| i)
            .chain([text.len()])
            .filter(|&i| {
                // never inside a special token
                !text[..i].ends_with("<")
                    && !text[..i].ends_with("<|")
                    && !text[..i].ends_with("<|\"")
                    && !text[..i].ends_with("<|\"|")
            })
            .collect();
        let mut last = String::new();
        for i in cuts {
            if let Some((_, g, _)) = call_json(&text[..i], true) {
                assert!(
                    fin.starts_with(&g),
                    "cut {i}: {g:?} not a prefix of {fin:?}"
                );
                assert!(g.starts_with(&last), "cut {i}: {g:?} shrank from {last:?}");
                last = g;
            }
        }
    }
}

/// The template writes a call's arguments with `format_argument`; reading
/// what it writes must give the arguments back, whatever JSON they are.
#[test]
fn what_the_template_writes_parses_back_to_the_same_arguments() {
    let cases = [
        json!({"file_path": "/src/main.rs", "limit": 200, "offset": 0}),
        json!({"content": "fn main() {\n    println!(\"hi, {}\", x);\n}\n", "file_path": "a.rs"}),
        json!({"edits": [{"old_string": "a: b", "new_string": "[c]"}], "replace_all": false}),
        json!({"query": "{nested: [1, 2]}", "filters": {"lang": ["en", "sv"], "min": -3.25}}),
        json!({"flag": true, "nothing": null, "count": 7}),
        json!({}),
    ];
    for args in cases {
        let messages = [
            json!({"role": "user", "content": "go"}),
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "tool_x", "arguments": args}}
            ]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "ok"}),
        ];
        let tools = [json!({"type": "function", "function": {"name": "tool_x",
            "description": "x", "parameters": {"type": "object", "properties": {}}}})];
        for (label, tpl) in TPLS {
            let text =
                crate::chat_template::render(tpl, &messages, Some(&tools[..]), None).unwrap();
            let s = text.find(CALL).expect("call rendered") + CALL.len();
            let e = text[s..].find(CALL_END).expect("call closed") + s;
            let (name, a, _) = call_json(&text[s..e], false).expect("parses");
            assert_eq!(name, "tool_x", "{label}");
            assert_eq!(
                serde_json::from_str::<Value>(&a).unwrap(),
                args,
                "{label}: {}",
                &text[s..e]
            );
            // and the result went in as a tool response the model can read
            assert!(
                text.contains(
                    "<|tool_response>response:tool_x{value:<|\"|>ok<|\"|>}<tool_response|>"
                ),
                "{label}"
            );
        }
    }
}

fn hints() -> crate::parsers::ToolHints {
    let tools = [
        json!({"type": "function", "function": {"name": "Read", "parameters": {
        "type": "object", "properties": {"file_path": {"type": "string"}}}}}),
    ];
    crate::parsers::tool_hints(Some(&tools[..])).unwrap()
}

#[test]
fn a_turn_splits_into_thought_calls_and_content() {
    let h = hints();
    let text = "<|channel>thought\nRead both.<channel|>\
                <|tool_call>call:Read{file_path:<|\"|>/a<|\"|>}<tool_call|>\
                <|tool_call>call:Read{file_path:<|\"|>/b<|\"|>}<tool_call|>";
    let p = parse(Dialect::GemmaChannel, text, false, Some(&h));
    assert_eq!(p.reasoning.as_deref(), Some("Read both."));
    assert_eq!(p.content, None);
    assert_eq!(p.tool_calls.len(), 2);
    assert_eq!(p.complete_calls, 2);
    assert_eq!(p.tool_calls[1].arguments, r#"{"file_path":"/b"}"#);
    assert_eq!(p.finish_reason(), "tool_calls");
    // a preamble stays content, a trailing <|tool_response> does not
    let p = parse(
        Dialect::GemmaChannel,
        "Let me look.<|tool_call>call:Read{file_path:<|\"|>/a<|\"|>}<tool_call|><|tool_response>",
        false,
        Some(&h),
    );
    assert_eq!(p.content.as_deref(), Some("Let me look."));
    assert_eq!(p.tool_calls.len(), 1);
    // no tools declared: the syntax is text
    let p = parse(Dialect::GemmaChannel, text, false, None);
    assert!(p.tool_calls.is_empty());
    assert!(p.content.unwrap().contains("<|tool_call>"));
}

#[test]
fn an_open_call_never_leaks_as_content() {
    let h = hints();
    for cut in [
        "<|tool_call>",
        "<|tool_call>call:Re",
        "<|tool_call>call:Read{file_pa",
    ] {
        let text = format!("Checking.{cut}");
        let p = parse(Dialect::GemmaChannel, &text, false, Some(&h));
        assert_eq!(p.content.as_deref(), Some("Checking."), "{cut}");
        assert_eq!(p.complete_calls, 0);
    }
    // a closed block that is no call stays text
    let p = parse(
        Dialect::GemmaChannel,
        "<|tool_call>not a call<tool_call|>",
        false,
        Some(&h),
    );
    assert!(p.tool_calls.is_empty());
    assert_eq!(
        p.content.as_deref(),
        Some("<|tool_call>not a call<tool_call|>")
    );
}

/// A call streams through tool_stream exactly like the final parse: started
/// once its name settles, its string argument growing as it is written.
#[test]
fn a_call_streams_its_arguments_while_written() {
    let h = hints();
    let text = "<|channel>thought\nok<channel|><|tool_call>call:Read{file_path:<|\"|>/very/long/path/to/a/file.rs<|\"|>}<tool_call|>";
    let mut ts = crate::tool_stream::ToolStream::default();
    let mut evs = Vec::new();
    let toks = [
        "<|channel>",
        "thought\nok",
        "<channel|>",
        "<|tool_call>",
        "call:Read{",
        "file_path:",
        "<|\"|>",
        "/very/long",
        "/path/to",
        "/a/file.rs",
        "<|\"|>",
        "}",
        "<tool_call|>",
    ];
    let mut prefix = String::new();
    for t in toks {
        prefix.push_str(t);
        let parsed = parse(Dialect::GemmaChannel, &prefix, false, Some(&h));
        evs.extend(ts.step(
            Dialect::GemmaChannel,
            &prefix,
            false,
            Some(&h),
            false,
            &parsed,
        ));
    }
    assert_eq!(prefix, text);
    let fin = parse(Dialect::GemmaChannel, text, false, Some(&h));
    evs.extend(ts.finish(&fin));
    use crate::tool_stream::ToolEv;
    let args: String = evs
        .iter()
        .filter_map(|e| match e {
            ToolEv::Args { json, .. } => Some(json.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(args, fin.tool_calls[0].arguments);
    let fragments = evs
        .iter()
        .filter(|e| matches!(e, ToolEv::Args { .. }))
        .count();
    assert!(fragments >= 4, "the path streamed in pieces: {evs:?}");
    assert!(matches!(evs.first(), Some(ToolEv::Start { name, .. }) if name == "Read"));
    assert!(matches!(evs.last(), Some(ToolEv::Stop { k: 0 })));
}

/// After tool results the model's own turn continues with no header. The
/// current template opens the thought there itself when thinking is on, the
/// older one writes nothing - the thought is open either way, and a
/// thinking-off turn answers directly.
#[test]
fn a_thinking_turn_goes_on_thinking_after_tool_results() {
    let messages = [
        json!({"role": "user", "content": "read it"}),
        json!({"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "read_file", "arguments": {"path": "/a"}}}]}),
        json!({"role": "tool", "tool_call_id": "c1", "content": "data"}),
    ];
    let tools = [
        json!({"type": "function", "function": {"name": "read_file", "description": "r",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}),
    ];
    for (label, tpl) in TPLS {
        for (think, open) in [(true, true), (false, false)] {
            let kw = json!({"enable_thinking": think});
            let p =
                crate::chat_template::render(tpl, &messages, Some(&tools[..]), Some(&kw)).unwrap();
            assert!(
                p.ends_with("<tool_response|>") || p.ends_with(crate::parsers::G_THOUGHT),
                "{label}: {p}"
            );
            assert_eq!(
                Dialect::GemmaChannel.thinking_open(&p),
                open,
                "{label}, thinking {think}"
            );
        }
        // a fresh turn: the bare header is thinking on, the pre-closed one off
        for (think, open) in [(true, true), (false, false)] {
            let kw = json!({"enable_thinking": think});
            let p = crate::chat_template::render(tpl, &messages[..1], None, Some(&kw)).unwrap();
            assert_eq!(
                Dialect::GemmaChannel.thinking_open(&p),
                open,
                "{label}, fresh, thinking {think}"
            );
        }
    }
}

/// A thinking-off turn continuing after tool results gets an empty closed
/// thought in both templates; a thinking one is opened instead (by the
/// current template, or by the pre-open over the older one); a fresh turn is
/// left to its own generation prompt.
#[test]
fn a_thinking_off_continuation_after_tool_results_gets_a_closed_thought() {
    let messages = [
        json!({"role": "user", "content": "read it"}),
        json!({"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "read_file", "arguments": {"path": "/a"}}}]}),
        json!({"role": "tool", "tool_call_id": "c1", "content": [
            {"type": "text", "text": "a.png contents:"}, {"type": "image"}]}),
    ];
    let closes = |p: &str| {
        !Dialect::GemmaChannel.thinking_open(p) && crate::parsers::g4_post_tool_unframed(p)
    };
    for (label, tpl) in TPLS {
        for think in [false, true] {
            let kw = json!({"enable_thinking": think});
            let p = crate::chat_template::render(tpl, &messages, None, Some(&kw)).unwrap();
            assert_eq!(closes(&p), !think, "{label}, thinking {think}: {p}");
        }
        let kw = json!({"enable_thinking": false});
        let fresh = crate::chat_template::render(tpl, &messages[..1], None, Some(&kw)).unwrap();
        assert!(!crate::parsers::g4_post_tool_unframed(&fresh), "{label}");
    }
}
