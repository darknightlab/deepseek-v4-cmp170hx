# Rebase notes

## Provenance

```text
haosdent head:       12810046c799cbe874967e19b1c0fa134ab7b209
haosdent merge-base: 62195e9784ebec1ece42b88a861734e0702cc2d5
official base:       402547d7f02bdbfc5dce5d27dc21f50dd4d627b6
```

`git rev-list --count <merge-base>..<haosdent-head>` returns `1`. That one
squash originally touched 111 files and mixed hardware enablement, performance
experiments, general behavior fixes, tests, and benchmarks.

## Why the squash was not kept as one patch

A direct cherry-pick automatically applied 93 files and conflicted in 18. The
first mechanical resolution kept the official side for every conflict, but the
result was only Git-clean, not source-consistent:

- the fork indexer caller expected query-sharding metadata omitted with the
  official `mla/indexer.py`;
- the fork MHC wrapper called kernels omitted with the official
  `tilelang_kernels.py`;
- several helper modules and environment switches remained after their model or
  worker callers had been replaced by official code.

Hardware execution later exposed additional paired contracts: SM80 DeepGEMM
fallbacks, FlashMLA planning, FP8 decode, disk O_DIRECT alignment, KV block
zeroing, and cudagraph dispatch. The final queue therefore separates required
support from optional optimization and removes unreachable providers.

## Layer 1: minimal SM80 correctness

The first patch contains only the path required to build and execute DSv4 on
pre-SM89 CUDA:

- Ampere backend selection and registry entries;
- Triton sparse MLA and MQA indexer logits;
- software E4M3 encode/decode for SM80;
- SM80-aware cache compression, indexer Q, and sparse decode;
- explicit prefill/decode dispatch when DeepGEMM is unavailable;
- MHC first-layer TileLang fallback;
- pre-Hopper CuTeDSL guards;
- ragged Triton metadata without FlashMLA scheduler planning;
- the official-base GDN declaration guard fix;
- three focused CUDA correctness tests.

The minimal MQA implementation uses conservative controls: ungrouped prefill,
no register cap, in-kernel Q decode, and no scale-factor hoist. It does not
modify `vllm/envs.py`.

## Layer 2: optional reachable performance work

The second patch can be omitted without disabling SM80 or downstream serving.
It contains fork code that still has a real caller:

- deterministic persistent top-k and sampler tie handling;
- Marlin occupancy experiments and optional FP8-to-BF16 dequantization;
- SM80 router GEMV;
- deterministic MoE alignment;
- hierarchical all-reduce and configurable custom-AR capacity;
- vocab-parallel local argmax infrastructure used by supported proposers;
- wide cache gather/dequant and compressor warp tuning;
- grouped/scaled indexer-logits tuning and paged-Q predecode;
- multi-stream safety and disable controls.

It deliberately excludes half-connected features:

```text
MHC int8 all-reduce and hoist
MHC fused-sqrsum/prenorm-shard providers without callers
DSpark fused Markov sampler without model hooks
attention GEMM unreplication/query sharding without metadata consumers
PinnedStagingPool without a model-runner consumer
cudagraph pad-up using the obsolete _is_compatible signature
```

Their modules and inert environment variables are not exported.

## Layer 3: downstream serving behavior

The third patch contains project-owned behavior rather than generic SM80
support:

- DSpark draft placement and propagation across pipeline ranks;
- long-context prefill top-k fallback and logits row chunking;
- structured-output and Responses fixes;
- DeepSeek reasoning/tool parser and tokenizer compatibility;
- worker response queue sizing;
- bounded disk KV offload with three independent physical/copy sizes:
  GPU block stride, payload bytes, and 4096-aligned disk stride.

The disk slot count is derived from the physical aligned stride, so configured
capacity remains bounded for non-aligned PP partitions.

## Conflict outcome

Of the original 18 conflict files, 12 remain byte-for-byte official. Six are
selectively modified on top of the official implementation:

```text
vllm/models/deepseek_v4/amd/rocm.py
vllm/models/deepseek_v4/attention.py
vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py
vllm/models/deepseek_v4/nvidia/model.py
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
vllm/v1/worker/gpu/model_runner.py
```

The first five carry required SM80 behavior. `gpu/model_runner.py` carries the
separate DSpark+PP downstream behavior.

Newer official KV block-zeroing fixes and their flat scheduler protocol are
retained as one unit. The fork's per-group scheduler half is not mixed with the
official worker half. The incompatible cudagraph pad-up experiment is also
omitted; official eager fallback is used for uncaptured shapes.

## Build and hardware findings

Full native build found and fixed:

- two out-of-scope `iter` references in persistent top-k, now `radix_iter`;
- `fused_gdn_decode_post_conv_mtp()` declared under KDA instead of GDN.

Three-card CMP 170HX startup found and fixed:

- DeepGEMM hard requirements before Triton fallback;
- missing first-layer MHC fallback;
- FlashMLA planning on the Ampere ragged Triton path;
- native `fp8e4nv` decode in shared ROCm/Ampere kernels;
- indexer prefill/decode wrappers still dispatching to DeepGEMM;
- fork-vs-official decode metadata field layout;
- O_DIRECT disk slots requiring naturally aligned PP payloads;
- mixed fork/official KV block-zeroing protocols;
- obsolete cudagraph compatibility call signature.

The final three-layer tree completed PP3 startup, sparse warmup, DSpark graph
capture, disk backend initialization, model listing, and a real chat request.

## Verification matrix

All three source combinations are checked from the pinned base:

```text
minimal only                  git am + compileall: PASS
minimal + downstream          git am + compileall: PASS
minimal + performance + downstream: PASS
full exported tree identity:  PASS
```

Performance/no-performance hardware comparison and four-card certification
remain outstanding.
