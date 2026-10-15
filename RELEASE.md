# Paddock 0.1.18

Mostly a Gemma 4 release. Gemma 4 can now call tools, so Claude Code and other agents can drive it, and it reads long prompts a lot faster. Also in this one: video embeddings on NVIDIA GPUs, Kumo Tabular past ten classes, faster k-quant kernels, and a RAM context cache that starts in seconds.

Builds for Windows x64, Linux x64 and the DGX Spark, on NVIDIA GPUs with driver 580 or newer. The macOS pre-release is built from the same commit.

## New

- **Gemma 4 tool calls** on chat completions, Messages and Responses. Calls stream as they're written, `tool_choice` works, and tool results (pictures too) go back to the model the way its template expects. Until now the calls came back as plain text.
- **EmbeddingGemma 2 video on NVIDIA GPUs** through `input_video`. Frames now use the 140 tokens the checkpoint asks for, on the Mac as well, where they were getting 70. A video that doesn't fit the 4,096-token context gets an error with its token count instead of being cut short.
- **Kumo Tabular up to 256 classes**, using NVIDIA's error-correcting output codes. With ten classes or fewer you get the same results as before.

## Improved

- Gemma 4 now speculates on requests that carry tools, which it didn't before, and the MoE model drafts 3 tokens instead of 7. Tool requests on the 26B-A4B with its MTP assistant went from 48.8 to 79.7 tokens/s on the DGX Spark.
- Gemma 4 reads long prompts faster: adding 4K tokens to a 150K conversation went from 805 to 1,374 tokens/s, and a cold 200K prompt from 1,130 to 1,833 (DGX Spark). Output is bit-identical.
- Gemma 4 keeps one copy of its embedding matrix instead of two: 0.78 GB less for the 26B-A4B and 1.5 GB less for the 31B at Q8_0.
- k-quant models (Q4_K, Q5_K, Q6_K, IQ4_XS, Q4_0): the prompt kernel is 10-17% faster, and checking speculative drafts uses a new tensor-core path. On an RTX 5090-class card, Qwen 3.8 27B at 4 bits under 12 Claude Code sessions went from 31.3 to 37.5 tokens/s per session. Output is bit-identical.
- The RAM context cache could take minutes to set up on some Linux hosts (a quarter of an hour for 32 GiB), and it dropped stores that arrived before it was ready. A 12 GiB cache now sets up in 6.5 s instead of 5 minutes, and early stores wait instead of getting dropped. With Qwen 3.8 27B under Claude Code, time to first token after the first turn went from 1.9-2.9 s to 0.75-0.9 s.
- Messages compaction got the same fix Responses got in 0.1.17: a span too big for the summarizer is done in parts instead of returning a 400, and streaming requests compact inside the stream with a ping every 15 s.
- A reasoning loop (the same span over and over, or a thought stuck on two or three words) now gets pushed on to the answer instead of running to the token limit. We hit this with Gemma 4 under Claude Code.
- A restore waiting for cache room now waits as long as recomputing the prompt would take before giving up.

## Fixed

- Gemma 4 thoughts no longer gain a newline every turn. An echoed thought came back one newline longer, the model copied that, and under Claude Code each tool step ended with one more newline and missed the prefix cache.
- A speculating Gemma 4 no longer logs that it can't speculate.
- A failed step on Qwen 3.5 to 3.8 could stop the engine, and later prompts on the same slots could get refused. Both fixed.
- README.txt in each archive now describes GPU support correctly and mentions the DGX Spark. It still talked about an RTX 40-series override that's long gone.

## macOS (pre-release)

- LightOnOCR-3 4B on MLX matches the reference on all six test pages, boxes included.
- EmbeddingGemma 2 video frames use 140 tokens.

## Known issues

- PaddleOCR-VL's layout pipeline is NVIDIA-only; on the Mac it reads whole pages.
- On the Mac, LightOnOCR-3 0.8B on MLX reads the same text as the reference, but a few boxes are off by up to about half a percent of the page.
- On the Mac, Nemotron on MLX matches the reference's first token on every test prompt, but not yet the full reply on all of them.
- SAM 3 and TIC Forestry are NVIDIA-only, and their downloads need a Hugging Face account that has accepted the model licence. TIC Forestry is made for Swedish aerial imagery at 0.5 m.
- Kolibri 1 can't be checked against llama.cpp yet, since no llama.cpp release reads it. We check it against Aleph Alpha's own serving code instead.
- EmbeddingGemma 2 audio clips are limited to 30 seconds.
- Laya is text-only, and its confidence isn't calibrated for more than ten options. The English checkpoint ships an out-of-range temperature for that case, which we clamp the same way Laya's own server does.
- On the NVFP4 versions of Qwen 3.8 Flash-Next, a greedy speculative reply can differ from plain decoding at a near tie. That's numerics, not a wrong answer.
- On-demand loading only covers Whisper so far.
