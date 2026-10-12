//! Server-side compaction for /v1/messages (`compact_20260112`): the
//! summarization pass, the agent loops' round-0 pass and the plain
//! two-iteration orchestration. A child of messages.rs (it shares the
//! parent's render/stream helpers); the pure halves - trigger math, span
//! split, resend rewrite - live in crate::context_management.

use super::*;

/// One summarization generation over the compact span of `messages`, shared
/// by the plain two-iteration path and the agent loops' round-0 pass.
struct SummaryPass {
    /// None = failed compaction (the model produced nothing usable).
    summary: Option<String>,
    /// This pass's `usage.iterations` entry, SDK shape.
    usage: Value,
    input_tokens: usize,
    output_tokens: usize,
    cached: usize,
}

/// Render [same system, same tools, span..., instructions] and generate the
/// summary. Same-prefix-as-the-conversation is the whole point: the
/// summarization prefill then rides the radix cache of the very conversation
/// being compacted. Err is a ready-to-return error response - including the
/// over-window case, where compaction cannot rescue the conversation and the
/// loud error stands (the Anthropic dialect has no fail-open backstop; its
/// Responses twin does, hence `summary_pass(lenient_gate)` there).
// Err is axum's own Response by design: a ready-to-return body, not a boxed error
#[allow(clippy::result_large_err)]
async fn anth_summary_pass(
    state: &Arc<AppState>,
    model: &ServingModel,
    req: &MessagesRequest,
    messages: &[Value],
    instructions: Option<&str>,
    scope: &crate::events::EventScope,
) -> Result<SummaryPass, Response> {
    let tail_start = crate::context_management::compact_tail_start(messages);
    let mut msgs1: Vec<Value> = messages[..tail_start].to_vec();
    let instructions =
        instructions.unwrap_or(crate::context_management::DEFAULT_COMPACT_INSTRUCTIONS);
    msgs1.push(json!({"role": "user", "content": instructions}));
    // same tools in the render (minus a tool_choice "none") - anything else
    // and the span prefix diverges from the live conversation's, losing the
    // radix hit that makes the summarization prefill nearly free
    let tools = if req
        .tool_choice
        .as_ref()
        .and_then(|tc| tc.get("type"))
        .and_then(Value::as_str)
        == Some("none")
    {
        None
    } else {
        match req.tools.as_ref().map(|ts| convert_tools(ts)).transpose() {
            Ok(t) => t,
            Err(e) => return Err(bad(e)),
        }
    };
    // tool removals from mid-conversation system messages too, folded over the
    // whole conversation exactly as the live render folds them - same reason
    let mut tools = tools;
    let controls = match crate::messages_system::system_controls(messages, req.tools.as_deref()) {
        Ok(c) => c,
        Err(e) => return Err(bad(e)),
    };
    crate::messages_system::drop_removed_tools(&mut tools, &controls.removed);
    // The live request's own thinking config and effort, not thinking off:
    // the template is NOT suffix-only there - Qwen3.8 writes its reasoning-
    // effort instruction into the SYSTEM block whenever thinking is on, so a
    // thinking-off render diverged a few tokens in and re-prefilled the whole
    // span (146K tokens, ~160 s, on an agent loop). An open think block is
    // closed at once by a one-token budget below, so the tokens still go to
    // the summary itself.
    let effort_owned = match parse_output_config(req.output_config.as_ref()) {
        Ok((e, _)) => e,
        Err(e) => return Err(bad(e)),
    };
    let effort = controls.effort.as_deref().or(effort_owned.as_deref());
    let (p1_ids, p1_engine, p1_think_open, p1_mm, _) = match render_prompt(
        model,
        req.system.as_ref(),
        &msgs1,
        tools.as_deref(),
        req.thinking.as_ref(),
        None,
        effort,
    ) {
        Ok(r) => r,
        Err(e) => return Err(bad(e)),
    };
    let close_think = crate::reasoning::summary_think_close(model, p1_think_open);
    // even the span + instructions overflows: compaction cannot rescue this
    // conversation, and the loud over-window error stands
    if let Some(e) =
        crate::chat::context_gate(model, p1_engine.len(), p1_mm.as_deref(), state.max_ctx)
    {
        return Err(engine_err(&e));
    }
    // greedy-class: a deterministic summary, no sampling surprises. The
    // truncation knobs still come from the model's election - at temperature
    // 0 they cannot change the argmax, so this is about staying consistent
    // with the rest of the server rather than about the draw.
    let dflt = state.sampling.resolve(true);
    let sampler1 = SamplingParams {
        temperature: 0.0,
        top_k: dflt.top_k,
        top_p: dflt.top_p,
        min_p: dflt.min_p,
        repeat_penalty: dflt.repeat_penalty,
        repeat_last_n: state.sampling.repeat_last_n,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: state.sampling.seed_or_now(req.seed),
        logit_bias: Vec::new(),
        no_repeat_ngram: (0, 0, false),
        reasoning: crate::reasoning::stop_policy(model, p1_think_open),
    };
    // bounded: a summary, not an essay (the pass is billed via usage.iterations)
    let max1 = state.max_output_ceiling.map_or(4096, |c| c.min(4096));
    let (tx, mut rx) = unbounded_channel();
    let gen1 = GenRequest {
        prompt: p1_engine,
        max_tokens: max1,
        sampler: sampler1,
        stop_tokens: model.stop_tokens.clone(),
        events: tx,
        mm_chunks: p1_mm,
        constraint: instantiate_constraint(
            &ConstraintSpec::None,
            GateSpec::Immediate,
            model,
            close_think.as_ref(),
        ),
        logprobs: None,
        submitted: None, // stamped by Engine::submit
        canvas_read: None,
        user_turn: false,
    };
    if let Err(e) = model.engine.submit(gen1) {
        return Err(err(StatusCode::INTERNAL_SERVER_ERROR, "api_error", e));
    }
    let mut ids1: Vec<u32> = Vec::new();
    let mut cached1 = 0usize;
    let mut p1_len = p1_ids.len();
    let mut terminal_tokens = 0;
    while let Some(evt) = next_event(&mut rx).await {
        match evt {
            TokenEvent::Prefilled { cached: c, rows } => {
                cached1 = c as usize;
                p1_len = p1_len.max(rows as usize);
            }
            TokenEvent::Token { id: t, .. } => ids1.push(t),
            TokenEvent::Done(_, stats) => {
                terminal_tokens = stats.terminal_tokens();
                scope.phases(&stats);
                break;
            }
            TokenEvent::Error(e) => return Err(engine_err(&e)),
        }
    }
    let raw = model.tokenizer.decode(&ids1, false).unwrap_or_default();
    let parsed = parse(model.dialect, &raw, p1_think_open, None);
    // None = failed compaction: the block reports null content (SDK pin) and
    // the request proceeds UNCOMPACTED - if that overflows, the loud
    // over-window error stands; never a silent trim
    let summary = parsed
        .content
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let mut usage = anth_usage(p1_len, cached1, ids1.len() + terminal_tokens);
    usage["type"] = json!("compaction");
    Ok(SummaryPass {
        summary,
        usage,
        input_tokens: p1_len,
        output_tokens: ids1.len() + terminal_tokens,
        cached: cached1,
    })
}

/// Round-0 context management for the MCP/web-search agent loops (
/// phase 5). The compact_20260112 orchestration runs once, here, on the
/// request's own prompt - the same measurement point a non-agent request uses
/// - and the compact edit is then dropped from the config so the loop's
///   per-round `prepare` cannot re-fire it. Three reasons that pin is right:
/// - the compact SPAN is fixed for the whole loop (the loop appends assistant
///   tool_use / user tool_result turns after the pending user message, which
///   is where the tail starts), so compacting at round 5 would remove exactly
///   what compacting here removes;
/// - the block then leads the content, which is where the spec'd resend
///   rewrite ("everything before the compaction block collapses into its
///   summary") is true of our summary - a block emitted mid-loop would tell
///   the client to drop tool round-trips the summary never saw;
/// - mid-turn tool bloat is `clear_tool_uses_20250919`'s job, and that one
///   does run every round.
///   Ok(Some) = compacted, `req.messages` replaced; Ok(None) = nothing to do;
///   Err = a ready-to-return response (an error, or the pause_after_compaction
///   answer, which is a complete response by definition).
// Err is axum's own Response by design: a ready-to-return body, not a boxed error
#[allow(clippy::result_large_err)]
pub(super) async fn precompact_agent(
    state: &Arc<AppState>,
    req: &mut MessagesRequest,
    scope: &crate::events::EventScope,
) -> Result<Option<CompactionOut>, Response> {
    if req.context_management.is_none() {
        return Ok(None);
    }
    let Some(model) = state.serving.as_ref() else {
        return Ok(None); // the loop reports "no model is loaded" in its own dialect
    };
    // one extra render+tokenize, and only for a request that configured
    // context management: the loop's round 0 re-prepares over the result
    let mut prepared =
        prepare(model, req, state.max_output_ceiling, &state.sampling).map_err(bad)?;
    let compact = prepared.compact.take();
    drop_compact_edit(&mut req.context_management);
    let Some(plan) = compact else { return Ok(None) };
    if plan.pause {
        // "compact and stop" - no tool can contribute to a summary, so the
        // plain orchestration answers this verbatim (stream and non-stream)
        let omit = thinking_omitted(req.thinking.as_ref()).unwrap_or(false);
        let cm = prepared.context_management.clone();
        return Err(
            run_compacting(state.clone(), req.clone(), plan, cm, omit, scope.clone()).await,
        );
    }
    let pass = anth_summary_pass(
        state,
        model,
        req,
        &plan.messages,
        plan.instructions.as_deref(),
        scope,
    )
    .await?;
    // a failed summary (null content) leaves the conversation uncompacted and
    // the block reports it - the loop still runs, and an over-window prompt
    // stays the loud error it always was
    req.messages = match &pass.summary {
        Some(s) => crate::context_management::compacted_messages(&plan.messages, s),
        None => plan.messages,
    };
    Ok(Some(CompactionOut {
        summary: pass.summary,
        usage: pass.usage,
    }))
}

/// Remove the compact edit from a config after the round-0 pass, so the agent
/// loops' per-round `prepare` cannot fire the trigger mid-turn (see the pin on
/// `precompact_agent`). The clear_* strategies stay: they are per-round by
/// design, and their `applied_edits` report still rides the final response.
pub(super) fn drop_compact_edit(cm: &mut Option<Value>) {
    if let Some(Value::Object(o)) = cm.as_mut()
        && let Some(Value::Array(edits)) = o.get_mut("edits")
    {
        edits.retain(|e| e.get("type").and_then(Value::as_str) != Some("compact_20260112"));
    }
}

/// `usage.iterations` for an agent turn that compacted: the summarization
/// pass, then the final message. Top-level usage stays the message's own, per
/// spec - iterations is where the extra generation is billed.
pub(super) fn agent_iterations(
    c: &CompactionOut,
    model_id: &str,
    prompt_len: usize,
    out_tokens: usize,
    cached: usize,
) -> Value {
    json!([
        c.usage,
        anth_iteration("message", model_id, prompt_len, cached, out_tokens)
    ])
}

/// The compact_20260112 orchestration (plan:
/// Two generations on the same
/// engine, sequentially: iteration 1 summarizes the compact span, iteration 2
/// is the real request over [framed summary + pending turn] - or, with
/// pause_after_compaction, the response stops at the block with
/// stop_reason "compaction". Cache economics are the design: iteration 1's
/// prefill rides the radix cache of the very conversation being compacted,
/// and the compacted conversation equals what a resend rewrites to, so every
/// later turn is a radix hit. Errors before the SSE stream starts return
/// clean HTTP errors even for stream=true (nothing has been committed yet).
pub(super) async fn run_compacting(
    state: Arc<AppState>,
    req: MessagesRequest,
    plan: CompactPlan,
    cm_report: Option<Value>,
    omit_thinking: bool,
    scope: crate::events::EventScope,
) -> Response {
    let Some(model) = state.serving.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded_error",
            "no model is loaded",
        );
    };

    // -- iteration 1: the summarization pass -----------------------------
    let pass = match anth_summary_pass(
        &state,
        model,
        &req,
        &plan.messages,
        plan.instructions.as_deref(),
        &scope,
    )
    .await
    {
        Ok(p) => p,
        Err(r) => return r,
    };
    let SummaryPass {
        summary,
        usage: usage1,
        input_tokens: p1_len,
        output_tokens: out1,
        cached: cached1,
    } = pass;

    // -- pause_after_compaction: the block is the response ---------------
    if plan.pause {
        scope.usage(p1_len, out1);
        scope.cached(cached1);
        scope.finish("compaction");
        let id = format!("msg_{}", uuid::Uuid::new_v4().simple());
        let block = compaction_block(&summary);
        if req.stream {
            let model_id = model.id.clone();
            let cm = cm_report.clone();
            let sse = stream! {
                yield ev("message_start", json!({"type": "message_start", "message": {
                    "id": id, "type": "message", "role": "assistant",
                    "model": model_id, "content": [],
                    "stop_reason": null, "stop_sequence": null,
                    "usage": {"input_tokens": p1_len, "output_tokens": 0},
                }}));
                yield ev("ping", json!({"type": "ping"}));
                yield ev("content_block_start", json!({
                    "type": "content_block_start", "index": 0,
                    "content_block": {"type": "compaction", "content": null}}));
                yield ev("content_block_delta", json!({
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "compaction_delta", "content": summary}}));
                yield ev("content_block_stop", json!({"type": "content_block_stop", "index": 0}));
                let mut usage = anth_usage(p1_len, cached1, out1);
                usage["iterations"] = json!([usage1]);
                let mut delta_ev = json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": "compaction", "stop_sequence": null},
                    "usage": usage,
                });
                if let Some(cm) = &cm {
                    delta_ev["context_management"] = cm.clone();
                }
                yield ev("message_delta", delta_ev);
                yield ev("message_stop", json!({"type": "message_stop"}));
            };
            return anth_sse(sse);
        }
        let mut usage = anth_usage(p1_len, cached1, out1);
        usage["iterations"] = json!([usage1]);
        let mut body = json!({
            "id": id, "type": "message", "role": "assistant", "model": model.id,
            "content": [block], "stop_reason": "compaction", "stop_sequence": null,
            "usage": usage,
        });
        if let Some(cm) = &cm_report {
            body["context_management"] = cm.clone();
        }
        return Json(body).into_response();
    }

    // -- iteration 2: the real generation over the compacted turn -------
    let msgs2 = match &summary {
        Some(s) => crate::context_management::compacted_messages(&plan.messages, s),
        None => plan.messages.clone(),
    };
    let mut req2 = req.clone();
    req2.messages = msgs2;
    req2.context_management = None;
    let prepared = match prepare(model, &req2, state.max_output_ceiling, &state.sampling) {
        Ok(p) => p,
        Err(e) => return bad(e),
    };
    if let Some(e) = crate::chat::context_gate(
        model,
        prepared.engine_prompt.len(),
        prepared.mm_chunks.as_deref(),
        state.max_ctx,
    ) {
        return engine_err(&e);
    }
    let ctx = Ctx {
        id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
        model_id: model.id.clone(),
        tokenizer: model.tokenizer.clone(),
        prompt_len: prepared.prompt_ids.len(),
        dialect: model.dialect,
        thinking_open: prepared.thinking_open,
        hints: prepared.hints.clone(),
        single_tool_call: prepared.single_tool_call,
        stop_strings: prepared.stop_strings.clone(),
        omit_thinking,
        context_management: cm_report,
        compaction: Some(CompactionOut {
            summary,
            usage: usage1,
        }),
        ocr: prepared.ocr.clone(),
        scope,
    };
    let constraint = instantiate_constraint(
        &prepared.constraint_spec,
        prepared.gate,
        model,
        prepared.think_budget.as_ref(),
    );
    let (tx, rx) = unbounded_channel();
    let gen2 = GenRequest {
        prompt: prepared.engine_prompt,
        max_tokens: prepared.max_tokens,
        sampler: prepared.sampler,
        stop_tokens: prepared.stop_tokens,
        events: tx,
        mm_chunks: prepared.mm_chunks,
        constraint,
        logprobs: None,
        submitted: None, // stamped by Engine::submit
        canvas_read: None,
        user_turn: false,
    };
    if let Err(e) = model.engine.submit(gen2) {
        return err(StatusCode::INTERNAL_SERVER_ERROR, "api_error", e);
    }
    if req.stream {
        stream_response(ctx, rx)
    } else {
        collect_response(ctx, rx).await
    }
}
