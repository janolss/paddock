use super::*;

/// CPU-only reconstruction of a captured benchmark's actual Anthropic adapter.
/// Counts must agree with the server before prefix-overlap conclusions are used.
#[test]
#[ignore = "requires PADDOCK_ANTHROPIC_CAPTURE and PADDOCK_TEMPLATE_MODEL"]
fn captured_anthropic_prefix_boundaries() {
    let model = std::path::PathBuf::from(std::env::var("PADDOCK_TEMPLATE_MODEL").unwrap());
    let capture = std::env::var("PADDOCK_ANTHROPIC_CAPTURE").unwrap();
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_hf_dir(&model).unwrap();
    let template = std::fs::read_to_string(model.join("chat_template.jinja")).unwrap();
    let rows: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(capture).unwrap()).unwrap();
    let mut previous: Vec<u32> = Vec::new();
    for (i, row) in rows
        .iter()
        .filter(|r| r["path"] == "/v1/messages")
        .enumerate()
    {
        let body = &row["effective"];
        let messages = convert_messages(
            body.get("system"),
            body["messages"].as_array().unwrap(),
            crate::chat_template::renders_late_system(&template),
            false,
        )
        .unwrap();
        let tools = body
            .get("tools")
            .map(|t| convert_tools(t.as_array().unwrap()).unwrap());
        let rendered = crate::chat_template::render(
            &template,
            &crate::chat_template::normalize_messages(&messages),
            tools.as_deref(),
            Some(&json!({"enable_thinking": false})),
        )
        .unwrap();
        let ids = tokenizer.encode(&rendered).unwrap();
        let usage = row["events"]
            .as_array()
            .unwrap()
            .iter()
            .rev()
            .find(|e| e["event"]["type"] == "message_delta")
            .unwrap()["event"]["usage"]
            .clone();
        let recorded: u64 = [
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ]
        .iter()
        .map(|k| usage[k].as_u64().unwrap_or(0))
        .sum();
        assert_eq!(
            ids.len() as u64,
            recorded,
            "offline token count, request {i}"
        );
        let common = previous
            .iter()
            .zip(&ids)
            .take_while(|(a, b)| a == b)
            .count();
        println!(
            "ANTHROPIC_PREFIX {}",
            json!({"request": i, "tokens": ids.len(),
            "cached": usage["cache_read_input_tokens"], "previous": previous.len(), "common": common,
            "old_cut_ids": &previous[common.saturating_sub(2)..previous.len().min(common+3)],
            "new_cut_ids": &ids[common.saturating_sub(2)..ids.len().min(common+3)],
            "old_cut": tokenizer.decode(&previous[common.saturating_sub(5)..previous.len().min(common+20)], false).unwrap(),
            "new_cut": tokenizer.decode(&ids[common.saturating_sub(5)..ids.len().min(common+20)], false).unwrap()})
        );
        previous = ids;
    }
}

#[test]
fn system_blocks_preserve_instruction_boundaries() {
    let system = json!([
        {"type": "text", "text": "# Agent policy"},
        {"type": "text", "text": "Run tests before editing.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Keep the final reply concise."}
    ]);
    for native in [false, true] {
        let messages = convert_messages(
            Some(&system),
            &[json!({"role": "user", "content": "Hi"})],
            native,
            false,
        )
        .unwrap();
        assert_eq!(
            messages[0],
            json!({"role": "system", "content":
                "# Agent policy\n\nRun tests before editing.\n\nKeep the final reply concise."})
        );
    }
}

#[test]
fn system_strings_and_non_system_fragments_are_not_rewritten() {
    let text = "  Keep this exact.\n\nIncluding whitespace.  ";
    let user = json!({"role": "user", "content": "Hi"});
    let messages = convert_messages(
        Some(&json!(text)),
        std::slice::from_ref(&user),
        false,
        false,
    )
    .unwrap();
    assert_eq!(messages[0]["content"], text);
    let messages = convert_messages(
        None,
        &[
            user.clone(),
            json!({"role": "assistant", "content":
            [{"type": "text", "text": "frag"}, {"type": "text", "text": "ment"}]}),
        ],
        false,
        false,
    )
    .unwrap();
    assert_eq!(messages[1]["content"], "fragment");
    let merged = merge_system(
        Some(json!([
            {"type": "text", "text": "Caller policy", "cache_control": {"type": "ephemeral"}}
        ])),
        "Server policy",
    );
    let messages = convert_messages(Some(&merged), &[user], false, false).unwrap();
    assert_eq!(messages[0]["content"], "Server policy\n\nCaller policy");
}

fn one_server_routing() -> HashMap<String, (paddock_mcp::ServerConfig, String)> {
    let cfg = paddock_mcp::ServerConfig {
        id: "test".into(),
        label: "artifacts".into(),
        transport: paddock_mcp::Transport::Http {
            url: "http://localhost/mcp".into(),
            headers: HashMap::new(),
        },
    };
    HashMap::from([(
        "artifacts__artifact_read".to_string(),
        (cfg, "artifacts".to_string()),
    )])
}

/// Lever 1 on the Anthropic lane: the same call twice comes off the
/// ledger, a third time is refused, and the wrapper spelling is the same
/// call as the direct one.
/// The Messages API's prompt split: `input_tokens` is the UNCACHED part
/// and the three fields sum to the prompt - the sum Claude Code sizes its
/// context (and its auto-compact trigger) from. A cached turn must not
/// count twice.
#[test]
fn usage_splits_the_prompt_by_cache() {
    let sum = |u: &Value| {
        [
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ]
        .iter()
        .map(|k| u[*k].as_u64().unwrap())
        .sum::<u64>()
    };
    let u = anth_usage(150_000, 149_000, 42);
    assert_eq!(u["input_tokens"], 1_000);
    assert_eq!(u["cache_read_input_tokens"], 149_000);
    assert_eq!(u["cache_creation_input_tokens"], 0);
    assert_eq!(u["output_tokens"], 42);
    assert_eq!(sum(&u), 150_000);
    // cold: all input, nothing cached
    let u = anth_usage(1_645, 0, 7);
    assert_eq!((u["input_tokens"].as_u64(), sum(&u)), (Some(1_645), 1_645));
    // a cache report can never exceed the prompt it came from
    assert_eq!(sum(&anth_usage(100, 250, 0)), 100);
    let it = anth_iteration("message", "m", 10, 4, 1);
    assert_eq!(
        (it["type"].as_str(), it["model"].as_str()),
        (Some("message"), Some("m"))
    );
    assert_eq!(sum(&it), 10);
}

#[test]
fn a_repeated_anthropic_call_replays_then_is_refused() {
    let routing = one_server_routing();
    let catalog = vec![crate::tool_search::CatalogTool {
        name: "artifacts__artifact_read".into(),
        description: "read one".into(),
        input_schema: json!({"type":"object","properties":{"artifact_id":{"type":"string"}},
                                 "required":["artifact_id"]}),
    }];
    let mut ledger = crate::loop_budget::CallLedger::new();
    let args = r#"{"artifact_id":"a1"}"#;

    let p = plan_anthropic_call(
        &routing,
        &catalog,
        &mut ledger,
        "artifacts__artifact_read",
        args,
    );
    assert!(matches!(p.action, AnthAction::Invoke { .. }));
    ledger.record(&p.sig.expect("a call that runs is filed"), true, "the page");

    // ...through the mcp_call_tool envelope this time: same call.
    let wrapped =
        r#"{"name":"artifacts__artifact_read","arguments_json":"{\"artifact_id\":\"a1\"}"}"#;
    let p = plan_anthropic_call(
        &routing,
        &catalog,
        &mut ledger,
        crate::tool_search::CALL_TOOL,
        wrapped,
    );
    match p.action {
        AnthAction::Replay { output } => assert!(output.ends_with("the page"), "{output}"),
        _ => panic!("the wrapper spelling must hit the ledger"),
    }
    assert_eq!(
        p.display_server, "artifacts",
        "a replay keeps the server on its card"
    );

    let p = plan_anthropic_call(
        &routing,
        &catalog,
        &mut ledger,
        "artifacts__artifact_read",
        args,
    );
    match p.action {
        AnthAction::Refuse { message } => assert!(message.contains("twice"), "{message}"),
        _ => panic!("the third emission is a loop"),
    }
}

/// The answer round takes our tools out of the rendered set and leaves the
/// caller's - a client tool call hands the turn back, which is a clean way
/// for an over-budget turn to end.
#[test]
fn the_answer_round_strips_our_tools_and_keeps_the_callers() {
    let routing = one_server_routing();
    let mut req: MessagesRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 16, "messages": [],
        "tools": [
            {"name": "get_weather", "input_schema": {"type": "object"}},
            {"name": "artifacts__artifact_read", "input_schema": {"type": "object"}},
            {"name": crate::tool_search::SEARCH_TOOL, "input_schema": {"type": "object"}},
            {"name": crate::tool_search::CALL_TOOL, "input_schema": {"type": "object"}},
        ],
    }))
    .expect("a valid request");
    strip_our_tools(&mut req, &routing, false, false, false, false);
    let names: Vec<&str> = req
        .tools
        .iter()
        .flatten()
        .filter_map(|t| t["name"].as_str())
        .collect();
    assert_eq!(names, ["get_weather"]);
}

/// A forcing tool_choice cannot survive the strip: `any` over an emptied
/// set, or `tool` naming an MCP tool that just left, both 400 the round.
#[test]
fn a_forcing_tool_choice_relaxes_with_the_tools() {
    let routing = one_server_routing();
    for forced in [
        json!({"type":"any"}),
        json!({"type":"tool","name":"artifacts__artifact_read"}),
    ] {
        let mut req: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 16, "messages": [], "tool_choice": forced,
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        }))
        .expect("a valid request");
        strip_our_tools(&mut req, &routing, false, false, false, false);
        assert_eq!(req.tool_choice, Some(json!({"type":"auto"})));
    }
    // "auto" and "none" are left exactly as the caller wrote them.
    let mut req: MessagesRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 16, "messages": [], "tool_choice": {"type":"none"},
    }))
    .expect("a valid request");
    strip_our_tools(&mut req, &routing, false, false, false, false);
    assert_eq!(req.tool_choice, Some(json!({"type":"none"})));
}

#[test]
fn stripping_the_only_tools_leaves_none_rather_than_an_empty_list() {
    let routing = one_server_routing();
    let mut req: MessagesRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 16, "messages": [],
        "tools": [{"name": "artifacts__artifact_read", "input_schema": {"type": "object"}}],
    }))
    .expect("a valid request");
    strip_our_tools(&mut req, &routing, false, false, false, false);
    assert!(
        req.tools.is_none(),
        "an empty tools array is not the same as no tools"
    );
}

/// The Anthropic path serves the same `{"type":"forensics"}` tool the
/// Responses path does: extraction pulls it out (gated on the runner's
/// config), the loop recognizes its calls, and the answer round strips it.
#[test]
fn anthropic_forensics_tool_extracts_gates_and_is_handled() {
    let rt = crate::forensics::ForensicRuntime::build(&crate::config::ForensicsConfig {
        enabled: true,
        auto: crate::config::ForensicsAuto::Off,
        tool: true,
        device: None,
    })
    .expect("runtime");

    // Enabled runner: the forensics entry is pulled out, the client tool stays.
    let mut state = crate::routes::AppState::for_tests(None);
    state.forensics = Some(rt);
    let mut req: MessagesRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 16, "messages": [],
        "tools": [
            {"name": "get_weather", "input_schema": {"type": "object"}},
            {"type": "forensics"},
        ],
    }))
    .expect("valid request");
    let got = extract_forensics_tool(&state, &mut req).expect("served");
    assert!(got.is_some(), "the runtime is returned when tool = true");
    let names: Vec<&str> = req
        .tools
        .iter()
        .flatten()
        .filter_map(|t| t["name"].as_str())
        .collect();
    assert_eq!(
        names,
        ["get_weather"],
        "forensics entry removed, client tool kept"
    );

    // The injected model-facing def + is_handled recognition.
    let def = crate::forensics::anthropic_tool_def();
    assert_eq!(def["name"], crate::forensics::TOOL_NAME);
    assert!(
        def["input_schema"].is_object(),
        "Anthropic def carries input_schema"
    );
    let routing = std::collections::HashMap::new();
    assert!(is_handled_call(
        crate::forensics::TOOL_NAME,
        &routing,
        false,
        false,
        true,
        false
    ));
    assert!(
        !is_handled_call(
            crate::forensics::TOOL_NAME,
            &routing,
            false,
            false,
            false,
            false
        ),
        "not handled when forensics is off"
    );

    // The answer round strips the forensics def (forensics_on).
    let mut req2: MessagesRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 16, "messages": [],
        "tools": [{"name": crate::forensics::TOOL_NAME, "input_schema": {"type": "object"}}],
    }))
    .expect("valid request");
    strip_our_tools(&mut req2, &routing, false, false, true, false);
    assert!(
        req2.tools.is_none(),
        "forensics def leaves the rendered set on the answer round"
    );

    // Disabled runner: requesting it is a clean 400, never a silent no-op.
    let mut off = crate::routes::AppState::for_tests(None);
    off.forensics = None;
    let mut req3: MessagesRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 16, "messages": [], "tools": [{"type": "forensics"}],
    }))
    .expect("valid request");
    assert!(
        extract_forensics_tool(&off, &mut req3).is_err(),
        "unset runner 400s"
    );

    // tool = false (auto-only runner) is also a 400 for the on-demand surface.
    let auto_only = crate::forensics::ForensicRuntime::build(&crate::config::ForensicsConfig {
        enabled: true,
        auto: crate::config::ForensicsAuto::Images,
        tool: false,
        device: None,
    })
    .expect("runtime");
    let mut auto_state = crate::routes::AppState::for_tests(None);
    auto_state.forensics = Some(auto_only);
    let mut req4: MessagesRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 16, "messages": [], "tools": [{"type": "forensics"}],
    }))
    .expect("valid request");
    assert!(
        extract_forensics_tool(&auto_state, &mut req4).is_err(),
        "tool=false 400s"
    );
}

/// The nudge joins the tool-result turn rather than opening a second user
/// turn: two user turns in a row is a shape Anthropic rejects and some
/// chat templates render badly.
#[test]
fn the_answer_nudge_joins_the_tool_result_turn() {
    let mut msgs = vec![
        json!({"role":"user","content":"go"}),
        json!({"role":"assistant","content":[{"type":"tool_use","id":"t1"}]}),
        json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"t1"}]}),
    ];
    append_user_text(&mut msgs, "answer now");
    assert_eq!(msgs.len(), 3, "no new turn");
    let blocks = msgs[2]["content"].as_array().expect("blocks");
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[1]["text"], "answer now");
}

#[test]
fn the_answer_nudge_opens_a_turn_when_the_last_one_is_the_assistant() {
    let mut msgs = vec![json!({"role":"assistant","content":"hm"})];
    append_user_text(&mut msgs, "answer now");
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[1]["role"], "user");
}

/// Claude Code ends every request with a system message. The nudge lands
/// before it - a user turn behind it would break its placement rule and
/// 400 the answer round.
#[test]
fn the_answer_nudge_keeps_a_trailing_system_message_last() {
    let sys = json!({"role":"system","content":"# Environment"});
    let mut msgs = vec![
        json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"t1"}]}),
        sys.clone(),
    ];
    append_user_text(&mut msgs, "answer now");
    assert_eq!(msgs.len(), 2, "joins the tool-result turn");
    assert_eq!(msgs[0]["content"][1]["text"], "answer now");
    assert_eq!(msgs[1], sys);

    let mut after_asst = vec![json!({"role":"assistant","content":"hm"}), sys.clone()];
    append_user_text(&mut after_asst, "answer now");
    assert_eq!(after_asst[1]["role"], "user");
    assert_eq!(after_asst[2], sys);
}

/// A Messages stream silent past the interval (a long cold prefill) sends
/// the spec's ping event - an SSE event Anthropic's SDKs parse and pass
/// over, not a comment - so Claude Code's idle watchdog sees traffic.
#[tokio::test]
async fn a_silent_messages_stream_pings() {
    use http_body_util::BodyExt;
    let mut body = anth_sse(futures::stream::pending()).into_body();
    let frame = tokio::time::timeout(std::time::Duration::from_secs(30), body.frame())
        .await
        .expect("a silent stream must not stay silent")
        .expect("stream stays open")
        .expect("body frame");
    let bytes = frame.into_data().expect("ping data");
    assert_eq!(
        std::str::from_utf8(&bytes).unwrap(),
        "event: ping\ndata: {\"type\": \"ping\"}\n\n"
    );
}

/// Claude Code's tool loop as it resends it: the model's thinking, a line
/// of text and a tool call, then the tool's result.
fn tool_loop_history() -> Vec<Value> {
    vec![
        json!({"role": "user", "content": "read the ledger"}),
        json!({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "PRIOR-THOUGHT: start with ledger_01", "signature": ""},
            {"type": "text", "text": "Reading it."},
            {"type": "tool_use", "id": "toolu_1", "name": "Read",
             "input": {"file_path": "ledger_01.txt"}},
        ]}),
        json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "REC 01-0001"},
        ]}),
    ]
}

/// A picture an agent's Read returned rides its tool reply into the
/// template as a picture slot (it used to be cut to the reply's text).
#[test]
fn a_read_picture_reaches_qwen38_in_its_tool_reply() {
    let qwen = include_str!("../tests/fixtures/qwen38_chat_template.jinja");
    let mut h = tool_loop_history();
    h[2]["content"][0]["content"] = json!([{"type": "image", "source": {"type": "base64",
            "media_type": "image/png", "data": "iVBORw0KGgo="}}]);
    let converted = convert_messages(None, &h, false, true).expect("convert");
    assert_eq!(find_images(&converted).expect("images").len(), 1);
    let msgs = crate::chat_template::normalize_messages(&converted);
    let out = crate::chat_template::render(qwen, &msgs, None, None).expect("render");
    assert!(
        out.contains("<tool_response>\n<|vision_start|><|image_pad|><|vision_end|>"),
        "{out}"
    );
    assert!(crate::chat_template::renders_tool_images(qwen));
    let gemma = include_str!("../tests/fixtures/gemma4_chat_template.jinja");
    assert!(!crate::chat_template::renders_tool_images(gemma));
}

/// An Anthropic caller's prior thinking has to reach the template the way
/// a chat-completions caller's `reasoning_content` does. It didn't: the
/// conversion wrote `thinking`, which only gpt-oss reads, so Qwen3.8 - whose
/// template keeps prior reasoning by default - was served Claude Code's
/// history with the model's own reasoning cut out, 11K tokens shorter than
/// llama.cpp's prompt for the same request.
#[test]
fn anthropic_thinking_renders_on_qwen38_like_chat_reasoning_content() {
    let qwen = include_str!("../tests/fixtures/qwen38_chat_template.jinja");
    let converted = convert_messages(None, &tool_loop_history(), false, false).expect("convert");
    let msgs = crate::chat_template::normalize_messages(&converted);
    let out = crate::chat_template::render(qwen, &msgs, None, None).expect("render");
    assert!(
        out.contains("<think>\nPRIOR-THOUGHT: start with ledger_01\n</think>"),
        "prior thinking dropped:\n{out}"
    );

    // byte-for-byte what a chat-completions caller gets for the same loop
    let chat = [
        json!({"role": "user", "content": "read the ledger"}),
        json!({"role": "assistant", "content": "Reading it.",
                   "reasoning_content": "PRIOR-THOUGHT: start with ledger_01",
                   "tool_calls": [{"id": "toolu_1", "type": "function", "function": {
                       "name": "Read", "arguments": {"file_path": "ledger_01.txt"}}}]}),
        json!({"role": "tool", "content": "REC 01-0001", "tool_call_id": "toolu_1"}),
    ];
    let chat_out = crate::chat_template::render(
        qwen,
        &crate::chat_template::normalize_messages(&chat),
        None,
        None,
    )
    .expect("render");
    assert_eq!(out, chat_out, "surfaces diverged");
}

/// gpt-oss reads `thinking`, and its Harmony template refuses `thinking`
/// beside `content` on a tool-calling turn - which is exactly what Claude
/// Code resends (thinking + text + tool_use), so the old conversion made
/// this history a 400 on gpt-oss. It renders now, with the turn's text as
/// its analysis (what the template does with content alone); a turn with
/// thinking and no text still carries its thinking to the template.
#[test]
fn anthropic_tool_loop_renders_on_gpt_oss() {
    let gptoss = include_str!("../tests/fixtures/gptoss_chat_template.jinja");
    let converted = convert_messages(None, &tool_loop_history(), false, false).expect("convert");
    let msgs = crate::chat_template::normalize_messages(&converted);
    let out = crate::chat_template::render(gptoss, &msgs, None, None).expect("render");
    assert!(
        out.contains("<|channel|>analysis<|message|>Reading it.<|end|>"),
        "rendered:\n{out}"
    );

    let mut quiet = tool_loop_history();
    quiet[1]["content"].as_array_mut().unwrap().remove(1); // drop the text block
    let converted = convert_messages(None, &quiet, false, false).expect("convert");
    let msgs = crate::chat_template::normalize_messages(&converted);
    let out = crate::chat_template::render(gptoss, &msgs, None, None).expect("render");
    assert!(
        out.contains("<|channel|>analysis<|message|>PRIOR-THOUGHT: start with ledger_01<|end|>"),
        "thinking dropped:\n{out}"
    );
}

/// The real granite-vision template, the same fixture the integration
/// test renders. Lives under tests/ because that is where the rest of the
/// template gates read it from; this is `cfg(test)` so it never ships.
fn granite_template() -> &'static str {
    include_str!("../tests/fixtures/granite_vision_chat_template.jinja")
}

/// This surface's image path end to end, short of the engine: an
/// Anthropic `{"type":"image","source":{base64,...}}` block through
/// `convert_messages` -> `normalize_messages` -> the granite template.
///
/// `convert_messages` is private, so this is the only place the Anthropic
/// conversion can be pinned against a real template. It has
/// been broken here before, and not subtly: with extraction ordered after
/// normalization the payload was destroyed and /v1/messages rejected every
/// image, because `source` is Anthropic's only inline image shape.
///
/// The assertion is equivalence with what an OpenAI client's `image_url`
/// part renders to - same picture, same question, same prompt.
#[test]
fn an_anthropic_image_block_renders_the_granite_prompt_the_chat_shape_does() {
    const URI: &str = "data:image/png;base64,iVBORw0KGgo=";
    let anthropic = [json!({
        "role": "user",
        "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
            {"type": "text", "text": "What is this?"}
        ]
    })];
    let converted = convert_messages(None, &anthropic, false, false).expect("convert");
    // the pixels have to still be reachable after conversion - this is the
    // ordering the surface got wrong (convert -> extract -> normalize)
    let urls = crate::chat::find_images(&converted).expect("find");
    assert_eq!(
        urls.iter().map(|r| r.url.as_ref()).collect::<Vec<_>>(),
        [URI]
    );

    let msgs = crate::chat_template::normalize_messages(&converted);
    let out = crate::chat_template::render(granite_template(), &msgs, None, None).expect("render");
    assert_eq!(out.matches("<image>").count(), 1, "rendered:\n{out}");
    assert!(
        out.contains("<image>\nWhat is this?"),
        "question dropped:\n{out}"
    );

    // byte-for-byte what an OpenAI client gets for the same picture
    let chat = [json!({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": URI}},
            {"type": "text", "text": "What is this?"}
        ]
    })];
    let chat_msgs = crate::chat_template::normalize_messages(&chat);
    let chat_out =
        crate::chat_template::render(granite_template(), &chat_msgs, None, None).expect("render");
    assert_eq!(out, chat_out, "surfaces diverged");
}

/// Task tags ride as ordinary message text, so a surface that mangles text
/// blocks silently disables the model's real interface rather than
/// erroring. Anthropic's text blocks convert 1:1, and this pins it.
#[test]
fn a_task_tag_still_expands_through_an_anthropic_text_block() {
    let anthropic = [json!({
        "role": "user",
        "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
            {"type": "text", "text": "<chart2csv>"}
        ]
    })];
    let converted = convert_messages(None, &anthropic, false, false).expect("convert");
    let msgs = crate::chat_template::normalize_messages(&converted);
    let out = crate::chat_template::render(granite_template(), &msgs, None, None).expect("render");
    assert!(
        out.contains("extract the data into a CSV table"),
        "tag did not expand:\n{out}"
    );
    assert!(
        !out.contains("<chart2csv>"),
        "tag survived into the prompt:\n{out}"
    );
}

#[test]
fn thinking_display_values() {
    assert!(!thinking_omitted(None).unwrap());
    assert!(!thinking_omitted(Some(&json!({"type": "enabled"}))).unwrap());
    assert!(!thinking_omitted(Some(&json!({"display": "summarized"}))).unwrap());
    assert!(thinking_omitted(Some(&json!({"display": "omitted"}))).unwrap());
    assert!(!thinking_omitted(Some(&json!({"display": "updates"}))).unwrap());
    assert!(thinking_omitted(Some(&json!({"display": "raw"}))).is_err());
}

#[test]
fn toolset_resolution_precedence() {
    // configs entry > default_config > system default (enabled, eager)
    let mut set = McpToolset::default();
    assert_eq!(set.resolve("t"), (true, false));
    set.default_enabled = Some(false);
    set.default_defer = Some(true);
    assert_eq!(set.resolve("t"), (false, true));
    set.tools.insert("t".into(), (Some(true), Some(false)));
    assert_eq!(set.resolve("t"), (true, false));
    assert_eq!(set.resolve("other"), (false, true));
}

#[test]
fn mcp_toolsets_extract_and_validate() {
    // toolsets pull out of `tools`; plain function tools stay
    let mut req: MessagesRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "mcp_servers": [{"type": "url", "url": "https://x", "name": "srv"}],
        "tools": [
            {"name": "fn_tool", "description": "d", "input_schema": {"type": "object"}},
            {"type": "mcp_toolset", "mcp_server_name": "srv",
             "default_config": {"enabled": false},
             "configs": {"a": {"enabled": true, "defer_loading": true}}},
        ],
    }))
    .unwrap();
    let sets = extract_mcp_toolsets(&mut req).unwrap();
    assert_eq!(req.tools.as_ref().unwrap().len(), 1);
    assert_eq!(sets["srv"].resolve("a"), (true, true));
    assert_eq!(sets["srv"].resolve("b"), (false, false));

    // an mcp_toolset without mcp_servers is a 400
    let mut bare: MessagesRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "mcp_toolset", "mcp_server_name": "srv"}],
    }))
    .unwrap();
    assert!(extract_mcp_toolsets(&mut bare).is_err());
}

#[test]
fn blocks_convert_to_chat_shapes() {
    let messages = vec![
        json!({"role": "user", "content": "hi"}),
        json!({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm", "signature": ""},
            {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"x": 1}},
        ]}),
        json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"},
            {"type": "text", "text": "thanks"},
        ]}),
    ];
    let out = convert_messages(Some(&json!("sys")), &messages, false, false).unwrap();
    assert_eq!(out[0]["role"], "system");
    assert_eq!(out[1]["content"], "hi");
    assert_eq!(out[2]["role"], "assistant");
    assert_eq!(out[2]["reasoning_content"], "hmm");
    assert_eq!(out[2]["tool_calls"][0]["function"]["name"], "f");
    // tool_result becomes its own tool message before the user text
    assert_eq!(out[3]["role"], "tool");
    assert_eq!(out[3]["tool_call_id"], "toolu_1");
    assert_eq!(out[4]["role"], "user");
    assert_eq!(out[4]["content"], "thanks");
}

/// The shape Claude Code sends (captured from 2.1.280): the trailing
/// system message used to 400 the whole request with `invalid message
/// role "system"`. It rides the user turn now, and the leading system
/// prompt - the front of every cached prefix - is untouched.
#[test]
fn a_trailing_system_message_rides_the_user_turn() {
    let messages = vec![
        json!({"role": "user", "content": [{"type": "text", "text": "Say hi"}]}),
        json!({"role": "system", "output_config": {"effort": "medium"}, "content": [
            {"type": "text", "text": "# Environment", "cache_control": {"type": "ephemeral"}},
        ]}),
    ];
    let out = convert_messages(Some(&json!("sys")), &messages, false, false).unwrap();
    assert_eq!(out.len(), 2, "no system turn past the first");
    assert_eq!(out[0]["content"], "sys");
    assert_eq!(out[1]["role"], "user");
    assert_eq!(
        out[1]["content"],
        "Say hi\n\n<system-reminder>\n# Environment\n</system-reminder>"
    );
}

/// Positional: after a tool round the reminder opens its own user turn
/// behind the results, and the history before it converts byte-for-byte
/// as it did without it - the radix prefix does not move.
#[test]
fn a_system_message_after_tool_results_keeps_the_prefix() {
    let head = vec![
        json!({"role": "user", "content": "go"}),
        json!({"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
        ]}),
        json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "42"},
        ]}),
    ];
    let mut with = head.clone();
    with.push(json!({"role": "system", "content": "tests are flaky here"}));
    let base = convert_messages(None, &head, false, false).unwrap();
    let out = convert_messages(None, &with, false, false).unwrap();
    assert_eq!(out[..base.len()], base[..]);
    assert_eq!(out.len(), base.len() + 1);
    assert_eq!(out[base.len()]["role"], "user");
}

/// Claude Code compacts on Anthropic's exact overflow wording - and only on
/// that, numbers included - so both overflow flavours speak it here, while
/// every other error keeps its own text.
#[test]
fn a_context_overflow_speaks_anthropics_words() {
    let want = "prompt is too long: 270000 tokens > 262144 maximum";
    let text = EngineError::context_overflow(270_000, 262_144);
    assert_eq!(anthropic_message(&text), want);
    let images = EngineError::context_overflow_images(270_000, 262_144, 200_000, 3);
    assert_eq!(anthropic_message(&images), want);
    assert_eq!(anthropic_message(&EngineError::invalid("nope")), "nope");
}

#[test]
fn a_misplaced_system_message_is_a_400_not_a_render() {
    let first = vec![
        json!({"role": "system", "content": "x"}),
        json!({"role": "user", "content": "hi"}),
    ];
    let e = convert_messages(None, &first, false, false).unwrap_err();
    assert!(e.contains("must follow a user message"), "{e}");
}

#[test]
fn image_blocks_become_data_uris() {
    let messages = vec![json!({"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/bmp", "data": "AAAA"}},
    ]})];
    let out = convert_messages(None, &messages, false, false).unwrap();
    assert!(out[0]["content"].is_array());
    let urls = crate::chat::find_images(&out).unwrap();
    assert_eq!(
        urls.iter().map(|r| r.url.as_ref()).collect::<Vec<_>>(),
        ["data:image/bmp;base64,AAAA"]
    );
}

#[test]
fn find_stop_picks_earliest() {
    let stops = vec!["xx".to_owned(), "b".to_owned()];
    assert_eq!(find_stop("a b xx", &stops), Some((2, "b")));
    assert_eq!(find_stop("nothing", &stops), None);
}

/// The agent loops measure the compact trigger once,
/// so the round-0 pass strips that edit - and only that edit, because the
/// clear_* strategies are per-round by design.
#[test]
fn the_compact_edit_is_stripped_after_round_zero() {
    let mut cm = Some(json!({"edits": [
        {"type": "clear_thinking_20251015"},
        {"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 800}},
        {"type": "clear_tool_uses_20250919"},
    ]}));
    compact::drop_compact_edit(&mut cm);
    let edits = cm.as_ref().unwrap()["edits"].as_array().unwrap();
    assert_eq!(edits.len(), 2);
    assert_eq!(edits[0]["type"], "clear_thinking_20251015");
    assert_eq!(edits[1]["type"], "clear_tool_uses_20250919");
    // idempotent, and a config without one is untouched
    compact::drop_compact_edit(&mut cm);
    assert_eq!(cm.as_ref().unwrap()["edits"].as_array().unwrap().len(), 2);
    let mut none: Option<Value> = None;
    compact::drop_compact_edit(&mut none);
    assert!(none.is_none());
}

#[test]
fn only_urls_the_conversation_already_showed_are_fetchable() {
    // This is web fetch's security guard, so it is tested on the shapes a
    // real conversation carries: prose, a nested tool result, and a search
    // result block. A model that could fetch a URL it INVENTED could
    // exfiltrate its own context by encoding it into a hostname.
    let convo = json!([
        {"role": "user", "content": "compare https://example.com/a and <https://b.test/x?q=1>."},
        {"role": "user", "content": [
            {"type": "text", "text": "also (https://c.test/page)"},
            {"type": "tool_result", "content": [{"type": "text", "text": "see https://d.test/deep/path"}]}
        ]}
    ]);
    let mut seen = std::collections::HashSet::new();
    harvest_urls(&convo, &mut seen);
    for u in [
        "https://example.com/a",
        "https://b.test/x?q=1",
        "https://c.test/page",
        "https://d.test/deep/path",
    ] {
        assert!(seen.contains(u), "{u} was not harvested from {seen:?}");
    }
    // the punctuation that ordinarily follows a URL in prose is not part
    // of it, and the trailing period must not become a different page
    assert!(!seen.contains("https://example.com/a."));
    assert!(!seen.contains("https://c.test/page)"));

    // a URL the model made up is refused however plausible it looks
    let invented = "https://example.com/a/../../secret";
    assert!(
        !seen.iter().any(|s| crate::websearch::same_url(s, invented)),
        "an unseen URL must not match"
    );
    // and the seen ones match under the forgiving comparison
    assert!(
        seen.iter()
            .any(|s| crate::websearch::same_url(s, "https://example.com/a/")),
        "a trailing slash is not a different page"
    );
}

#[test]
fn a_conversation_with_no_links_makes_nothing_fetchable() {
    let convo = json!([{"role": "user", "content": "no links here, just httpish words"}]);
    let mut seen = std::collections::HashSet::new();
    harvest_urls(&convo, &mut seen);
    assert!(seen.is_empty(), "{seen:?}");
}

/// `parse_output_config` splits Anthropic's `{effort, format}` into the two
/// things this server acts on. The HTTP probes in tests/param_surface.rs
/// only prove a shape is accepted or refused - a mutation that parsed the
/// effort and then DISCARDED it passed every one of them - so the parse
/// gets checked on its output here.
///
/// The remaining untested link is the one-line call site that folds the
/// parsed rung into the template kwargs: that needs a loaded model, so it
/// is covered by the model-bearing SDK gate and by nothing cheaper.
mod output_config {
    use super::super::parse_output_config;
    use serde_json::json;

    #[test]
    fn splits_both_halves_and_keeps_the_schema_verbatim() {
        let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let cfg = json!({"effort": "xhigh", "format": {"type": "json_schema", "schema": schema}});
        let (effort, format) = parse_output_config(Some(&cfg)).expect("ok");
        assert_eq!(effort.as_deref(), Some("xhigh"));
        // the schema is lifted out of `format`, not left wrapped: Anthropic
        // puts it directly on `format` where OpenAI nests it one deeper
        assert_eq!(format.expect("format"), schema);
    }

    #[test]
    fn absent_and_empty_are_both_nothing() {
        assert_eq!(parse_output_config(None).expect("ok"), (None, None));
        assert_eq!(
            parse_output_config(Some(&json!({}))).expect("ok"),
            (None, None)
        );
        // explicit nulls are the SDK's way of spelling "unset"
        let cfg = json!({"effort": null, "format": null});
        assert_eq!(parse_output_config(Some(&cfg)).expect("ok"), (None, None));
    }

    #[test]
    fn every_malformed_shape_names_itself() {
        for (cfg, want) in [
            (json!("high"), "must be an object"),
            (json!({"effrot": "high"}), "unsupported output_config field"),
            (json!({"effort": 3}), "must be a string"),
            (
                json!({"format": {"type": "regex"}}),
                "unsupported output_config.format",
            ),
            (json!({"format": {"schema": {}}}), "needs a `type`"),
            (
                json!({"format": {"type": "json_schema"}}),
                "schema is required",
            ),
        ] {
            let e = parse_output_config(Some(&cfg)).expect_err("should refuse");
            assert!(e.contains(want), "expected {want:?} for {cfg}, got {e}");
        }
    }

    #[test]
    fn the_five_published_rungs_all_parse() {
        // Anthropic publishes low|medium|high|xhigh|max; every one is
        // inside the seven `reasoning_effort_rank` already takes, which is
        // why adopting their ladder needed no new vocabulary.
        for e in ["low", "medium", "high", "xhigh", "max"] {
            let cfg = json!({"effort": e});
            let (got, _) = parse_output_config(Some(&cfg)).expect("ok");
            assert_eq!(got.as_deref(), Some(e));
            assert!(
                crate::chat::reasoning_effort_rank_is_valid(e),
                "{e} unknown to the ladder"
            );
        }
    }
}
