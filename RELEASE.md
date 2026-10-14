# Paddock 0.1.18

A Gemma 4 release: Gemma 4 calls tools, speculates on agent requests and
reads long prompts far faster. EmbeddingGemma 2 embeds video on NVIDIA GPUs,
Kumo Tabular classifies up to 256 classes, k-quant models read prompts and
check drafts faster, and the RAM context cache starts in seconds and keeps
what it stores. Windows x64, Linux x64 and the NVIDIA DGX Spark. NVIDIA
GPUs, driver 580 or newer. The macOS pre-release for Apple Silicon is built
from the same commit.

## New

- **Gemma 4 calls tools** on chat completions, Messages and Responses. Its
  calls are read from Gemma's own call syntax, streamed while they are
  written, forced when a request requires a tool, and the turn carries on
  correctly after the tool results, pictures in them included. Claude Code
  and other agents can drive it now; before, its calls came back as plain
  text.

- **EmbeddingGemma 2 embeds video on NVIDIA GPUs** (`input_video`), as it
  already did on the Mac. Each frame now takes the 140 tokens the checkpoint
  is configured for - on the Mac too, where frames took half that - and a
  video too long for the 4,096-token context is refused with its token
  count instead of being cut.

- **Kumo Tabular classifies up to 256 classes**, with NVIDIA's own
  error-correcting output codes: the model runs once per code task and the
  results combine, as NVIDIA's pipeline does. Tables of ten classes or fewer
  give exactly the results they gave before.

## Improved

- **Gemma 4 speculates on agent requests.** Requests that declare tools
  never speculated; now they do, and the MoE model drafts 3 tokens ahead
  instead of 7. On the DGX Spark, tool requests on the 26B-A4B with its MTP
  assistant decode at 79.7 tokens a second instead of 48.8.

- **Gemma 4 reads long prompts far faster.** Its global attention layers
  load the next tile while they compute. On the DGX Spark a 4K-token
  addition to a 150K conversation reads at 1,374 tokens a second instead of
  805, and a cold 200K prompt at 1,833 instead of 1,130. The results are the
  same to the bit.

- **Gemma 4 holds its embedding matrix once instead of twice:** 0.78 GB
  less for the 26B-A4B and 1.5 GB less for the 31B at Q8_0.

- **Faster k-quant models** (Q4_K, Q5_K, Q6_K, IQ4_XS, Q4_0). The kernel
  that reads prompts is 10-17% faster, and the small batches that check
  speculative drafts run on a new tensor-core path: under a 12-session
  Claude Code load on Qwen 3.8 27B at 4 bits, decode went from 31.3 to 37.5
  tokens a second per session on an RTX 5090-class card. The results are
  the same to the bit.

- **The RAM context cache starts in seconds and keeps what it stores.**
  Setting up its memory could take minutes on a busy Linux host - a 32 GiB
  cache a quarter of an hour - and while it was being set up, stores that
  did not fit yet were dropped, so prompts came back half cached. A 12 GiB
  cache now sets up in 6.5 seconds instead of 5 minutes, and a store waits
  for room instead of being dropped: under a Claude Code load on Qwen 3.8
  27B the time to first token after the first turn fell from 1.9-2.9 s to
  0.75-0.9 s.

- **Messages compaction works like the Responses API's did in 0.1.17.** A
  span too big for the summarizer is summarized in parts instead of failing
  with a 400, and a streaming request compacts inside the stream with a
  ping every 15 seconds.

- **A reasoning loop ends.** A thought that repeats a span over and over, or
  collapses onto two or three words, is moved on to its answer instead of
  running to the token limit. Found with Gemma 4 under Claude Code.

- **A restore waiting for room waits as long as reading the prompt again
  would take** before it gives up, so giving up never costs more time than
  it saves.

## Fixed

- **Gemma 4's thoughts no longer grow a newline every turn.** A thought a
  client echoed back re-rendered one newline longer than the model wrote it,
  and the model copied the longer run: under Claude Code each tool step's
  thought ended in one more newline, and the history missed the prefix
  cache at every thought.

- **A speculating Gemma 4 no longer logs that it cannot speculate.**

- **One failed step on Qwen 3.5 to 3.8 could stop the engine.** A step that
  failed after its prompt work had started left that work marked in flight,
  and the next step stopped the engine; failure paths also left dead
  prompts queued, so later prompts on those slots were refused.

- **The README in every archive describes GPU support as the engine serves
  it** and names the DGX Spark; it still described an override for RTX
  40-series cards that no longer exists.

## macOS (pre-release)

- LightOnOCR-3 4B's MLX build matches the reference on all six test pages,
  grounding coordinates included.
- EmbeddingGemma 2 video frames take the checkpoint's 140 tokens.

## Known

- **PaddleOCR-VL's layout pipeline runs on NVIDIA GPUs only;** on the Mac
  it reads whole pages.

- **On the Mac, LightOnOCR-3 0.8B's MLX build reads the same text as the
  reference on every test page, but a few grounding boxes sit up to about
  half a percent of the page away from the reference's.** The 4B now
  matches the reference exactly.

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

- **EmbeddingGemma 2 takes audio clips of up to 30 seconds.**

- **Laya reads text only**, and its confidence on choices with more than ten
  options is not calibrated: the English checkpoint ships an out-of-range
  temperature for that case, which Paddock clamps, as Laya's own server does.

- **On Qwen 3.8 Flash-Next's NVFP4 versions a greedy speculative reply can
  part from plain decoding at a near tie** - the same model with slightly
  different numerics, not a wrong answer.

- On-demand loading covers Whisper only.
