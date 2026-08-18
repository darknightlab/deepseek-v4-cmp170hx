# Rebase and feature-port notes

## Provenance

```text
fork head:       12810046c799cbe874967e19b1c0fa134ab7b209
fork merge-base: 62195e9784ebec1ece42b88a861734e0702cc2d5
official pin:    402547d7f02bdbfc5dce5d27dc21f50dd4d627b6
```

The fork is one squash commit ahead of its merge-base. That commit mixes SM80
enablement, adopted optimization, measured regressions, correctness backports,
API behavior, tests, and benchmarks across 111 files. A direct cherry-pick onto
the official pin applies 93 files and conflicts in 18.

## Why every optimization is separate

The original mechanical rebase left providers without consumers: MHC int8,
fused Markov, query sharding, pinned staging, and multiple environment switches
were present even though their model/worker callers had been replaced with
official code. A clean `git am` therefore did not imply a complete feature.

The current queue uses one commit per feature contract. Caller, metadata,
kernel, environment switch, and tests travel together. Features that are not
independently hardware-certified are `experiment(...)` patches and are omitted
by the default profile.

## Minimal correctness patch

`0001` contains only the pre-SM89 execution path:

- Ampere backend selection;
- Triton sparse MLA and indexer MQA logits;
- software E4M3 encode/decode;
- explicit DeepGEMM/CuTeDSL/MHC fallbacks;
- ragged Triton metadata without FlashMLA planning;
- the official-pin GDN declaration guard fix;
- focused CUDA tests.

Its MQA controls are conservative: KV group 1, no register cap, in-kernel Q
decode, and no k-scale factoring. It does not modify `vllm/envs.py`.

## SM8x deployment profile

`VLLM_FORK_PERFORMANCE_PROFILE=sm8x` applies only the minimal correctness
patch, SM80/SM86-specific tuning (`0002`, `0004`-`0006`, `0008`, `0013`,
`0014`), and downstream `0023`. It excludes architecture-neutral and unrelated
experimental features. The 170HX compose must still set
`VLLM_MARLIN_FP8_DEQUANT_BF16=1`; applying `0004` provides the path but does
not force its memory/performance trade globally.

## Verified feature patches

The default verified profile explicitly applies `0002`-`0005` and
`0007`-`0015`; `0006` is SM8x-specific but remains current-port experimental.
The patches cover indexer logits, top-k, Marlin-dequant, router GEMV,
deterministic MoE alignment, all-reduce, local argmax, cache gather, compressor
warps, and multi-stream control.

## Experimental feature patches

- `0006`: Attention indexer-weights SM80 GEMV; included by `sm8x`, excluded
  from the default verified profile until current-port hardware A/B.
- `0016`: the fork's Marlin occupancy/warp perturbations. The measurement record
  says these regress; flags remain off.
- `0017`: cudagraph pad-up ported to the current six-argument
  `_is_compatible(..., max_query_len)` API. The obsolete five-argument version
  caused a production EngineCore failure and is not exported.
- `0018`: pinned staging pool plus all three model-runner consumers.
- `0019`: adaptive Marlin MoE block-size API.
- `0020`: persistent Marlin MoE workspace wired into all current call sites.
- `0021`: Attention input projection fusion only.
- `0022`: replicated Attention GEMM token sharding only.
- `0024`: DSpark vocab-sharded Markov selection while retaining official
  confidence/adaptive-verification behavior.
- `0025`: fused Markov kernels; automatically declines when adaptive verification
  is enabled or fusion operands are unsupported.
- `0026`: rank-uniform indexer prefill/decode query sharding with caller,
  metadata, and attention Q-path wiring in one patch.

## Downstream patch

`0023` contains project behavior rather than generic fork optimization:

- DSpark placement and draft propagation over PP;
- long-context top-k fallback and row chunking;
- parser, Responses, and structured-output fixes;
- bounded O_DIRECT disk KV with separate GPU stride, payload size, and
  4096-aligned disk stride.

It intentionally retains the official KV block-zeroing protocol as one unit.
Mixing the fork scheduler half with the official worker half caused first-request
failure and is not exported.

## Remaining semantic ports

Ten effective fork optimizations remain unmerged: four MHC features and six
Sparse-MLA features.

### MHC group

The fork MHC changes span CUDA custom all-reduce, stable-ABI declarations,
TileLang kernels, attention/MoE all-reduce ownership, decoder return arity, and
DSpark. The captured-data record also identifies accumulation-order changes and
measured regressions. The raw fork hunk cannot be an independent patch on the
pin. It must be split into at least:

1. fused post/sqrsum;
2. prenorm row sharding;
3. fixed split selection;
4. int8 all-reduce and its complete ownership contract.

### Sparse-MLA group

The remaining campaign changes alter both ROCm/Ampere kernels and their graph
metadata. They must be ported as independent features:

1. LUT-based FP8 decode;
2. 8192-row ragged prefix scan;
3. ratio-128 query-blocked prefill;
4. uniform decode grouping;
5. decode split tuning;
6. exact-tile specialization.

The decode-maxnreg and query-blocked-decode knobs are preserved in the fork
record as measured regressions and are not counted among these ten effective
optimizations.

The existing `0011`, `0013`, and `0026` already carry the independently
portable cache, indexer-logits, and query-sharding parts.

No non-applying or compile-broken source diff is stored as a usable patch.

## Hardware findings already folded into the queue

- persistent top-k used out-of-scope `iter` instead of `radix_iter`;
- the GDN declaration was guarded by KDA;
- constructor and forward paths still hard-required DeepGEMM;
- first-layer MHC lacked its TileLang fallback;
- Ampere ragged decode attempted FlashMLA planning;
- shared sparse kernels used native `fp8e4nv` on SM80;
- disk slots assumed naturally 4096-aligned PP payloads;
- fork/official KV zeroing protocols were mixed;
- cudagraph pad-up used an obsolete compatibility signature.

## Verification matrix

```text
minimal + downstream profile:       git am + compileall PASS
verified + downstream profile:      git am + compileall PASS
all currently exported patches:     git am + compileall PASS
all-profile submitted tree identity: PASS
```

Compile success is not hardware certification. Experimental patches require
individual A/B and correctness gates before production enablement.
