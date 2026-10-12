# Paddock 0.1.17

An agent-sessions release: thinking replies no longer stop dead when the
model samples its end token in the middle of its reasoning, agent loops on
the Responses API keep their reasoning across tool steps and compact turns
that outgrow the context, and the RAM context cache no longer stalls every
stream while it fills. Windows x64, Linux x64 and the NVIDIA DGX Spark.
NVIDIA GPUs, driver 580 or newer. The macOS pre-release for Apple Silicon
is built from the same commit.

## Improved

- **Agent loops on the Responses API keep their reasoning.** A reasoning
  item followed by tool calls was dropped, and every tool call went back as
  its own assistant message, so between tool steps the model lost its own
  chain of thought and read its previous reply again. One response's
  output now goes back as one assistant turn. Found driving the OpenAI
  Agents SDK through long sessions.

- **Responses compaction works the way OpenAI's does.** The compaction item
  stands for everything before it, user messages stay word for word, and a
  single long agent turn compacts too. A turn too big for the summarizer
  to read at once - an agent reading 14 files in one round sent 294K
  tokens to a 262K model - is now summarized in parts instead of failing
  with a 400. A streaming request compacts inside the stream and sends a
  keep-alive every 15 seconds, so clients that give up on a silent stream
  stay connected.

- **The RAM context cache no longer stalls streams while it fills.** Its
  memory was set up a GiB at a time on the serving thread, and each GiB
  paused every stream for about a second: on the DGX Spark an agent
  session on Qwen 3.8 ran at 10-14 tokens a second instead of 40-60. The
  cache now sets up its whole RAM allowance in the background as soon as
  it starts, and on Linux in huge pages, about five times faster.

- **Completions take token-id prompts.** `/v1/completions` accepts `prompt`
  as an array of token ids, or an array of such arrays, as the OpenAI
  schema allows. The ids go to the model exactly as sent, each checked
  against the vocabulary.

## Fixed

- **Thinking replies finish their thought.** Reasoning models, Qwen 3.8
  most of all, sometimes sample their end token in the middle of their
  reasoning, which ended the reply with no answer - under Claude Code a
  compaction then reported "summarization produced empty response". Inside
  an open reasoning block a stop is now masked before the draw, and a model
  that keeps reaching for it (three times within 64 tokens) is moved on to
  its answer instead of thinking to the token limit. On NVIDIA GPUs the
  mask runs on the GPU, so thinking replies keep their speed. On Qwen,
  Laguna and Gemma 4.

- **A request cut off by a shutdown ends with a retryable error** ("the
  server is shutting down - retry the request") instead of looking like a
  finished reply. When a runner stopped mid-request, for a model switch or
  a restart, Claude Code kept the cut-off reply as complete.

- **A system prompt sent as several blocks keeps its boundaries** on the
  Messages API. Claude Code and the Agent SDK send it that way, and the
  blocks were run together, which could fuse words or pull the next
  instruction into a heading or list item. They are now joined with a
  blank line.

## macOS (pre-release)

- Nemotron 3.5 Lightning serves from an MLX 8-bit package, downloadable
  from the catalog, and cached agent sessions on it run faster.
- Faster Qwen 3.8 Flash Next: expert routing and staging.
- The context cache keeps its measured restore costs across restarts,
  keeps Qwen's user-turn checkpoints through long tool loops without more
  GPU memory, and keeps Nemotron's prefixes.

## Known

- **PaddleOCR-VL's layout pipeline runs on NVIDIA GPUs only;** on the Mac
  it reads whole pages.

- **On the Mac, LightOnOCR-3's MLX build reads the same text as the
  reference on every test page, but a few grounding boxes sit up to about
  half a percent of the page away from the reference's.**

- **On the Mac, Nemotron's MLX build matches the reference's first token on
  every test prompt but not yet its whole reply on all of them;** full
  parity is still open.

- **SAM 3 and TIC Forestry run on NVIDIA GPUs only**, and their weights
  download only with a Hugging Face account that has accepted the model's
  licence. TIC Forestry is built for Swedish imagery at 0.5 m; it is not a
  general-purpose aerial model.

- **Kolibri 1 has no same-weights llama.cpp reference yet:** no llama.cpp
  release reads it, so its correctness is checked against Aleph Alpha's own
  serving code.

- **EmbeddingGemma 2 takes audio clips of up to 30 seconds;** on NVIDIA GPUs
  video is not served yet.

- **Laya reads text only**, and its confidence on choices with more than ten
  options is not calibrated: the English checkpoint ships an out-of-range
  temperature for that case, which Paddock clamps, as Laya's own server does.

- **Kumo Tabular predicts up to ten classes.**

- **On Qwen 3.8 Flash-Next's NVFP4 versions a greedy speculative reply can
  part from plain decoding at a near tie** - the same model with slightly
  different numerics, not a wrong answer.

- On-demand loading covers Whisper only.

- The fp8 KV cache's paged attention on RTX 50-series cards shows a small
  numeric deviation in one split configuration (#5).
