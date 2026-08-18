# Official upstream patch queue

Pinned base:

```text
vllm-project/vllm 402547d7f02bdbfc5dce5d27dc21f50dd4d627b6
```

Patches are grouped by purpose. The order for each build profile is explicit in
`patches/series/*.txt`; directory order is never used as an implicit dependency
order.

## Profiles

```bash
# Default: SM80/SM86 correctness and hardware-specific tuning + downstream
./scripts/prepare-upstream-vllm.sh /opt/vllm-sm8x

# SM8x baseline plus isolated MHC-prenorm and sparse-prefill experiments
VLLM_FORK_PERFORMANCE_PROFILE=sm8x-perf \
  ./scripts/prepare-upstream-vllm.sh /opt/vllm-sm8x-perf

# Minimal SM8x execution + downstream, no performance patches
VLLM_FORK_PERFORMANCE_PROFILE=none \
  ./scripts/prepare-upstream-vllm.sh /opt/vllm-minimal

# SM8x and architecture-neutral verified patches + downstream
VLLM_FORK_PERFORMANCE_PROFILE=verified \
  ./scripts/prepare-upstream-vllm.sh /opt/vllm-verified

# Every exported experimental port
VLLM_FORK_PERFORMANCE_PROFILE=all \
  ./scripts/prepare-upstream-vllm.sh /opt/vllm-all
```

| Profile | Series file | Purpose |
|---|---|---|
| `none` | `series/none.txt` | Minimal SM8x path plus downstream |
| `sm8x` | `series/sm8x.txt` | Default SM80/SM86-only tuning plus downstream |
| `sm8x-perf` | `series/sm8x-perf.txt` | `sm8x` plus only MHC prenorm cuBLAS and sparse-prefill BLOCK_K=32 |
| `verified` | `series/verified.txt` | SM8x and general verified features |
| `all` | `series/all.txt` | Every compile-checked experiment |

The 170HX runtime must still set:

```text
VLLM_MARLIN_FP8_DEQUANT_BF16=1
```

The SM8x series includes the implementation patch, but does not force this
VRAM/performance trade for every deployment.

## Categories

### `patches/core/`

| Patch | Feature |
|---|---|
| `0001-sm8x-correctness.patch` | Ampere backend, software FP8, Triton sparse MLA/MQA, DeepGEMM/CuTeDSL/MHC fallbacks |

### `patches/sm8x/`

| Patch | Feature |
|---|---|
| `0001-indexer-mqa.patch` | SM80 MQA/indexer logits tuning |
| `0002-marlin-fp8-dequant.patch` | Optional FP8-to-BF16/cuBLAS loading path |
| `0003-router-gemv.patch` | SM80 router BF16 GEMV |
| `0004-custom-ar-crossover.patch` | A100 custom-AR one-shot crossover |
| `0005-sparse-kv-gather.patch` | Sparse KV gather/dequant tuning |
| `0006-compressor-warps.patch` | Sparse compressor warp sizing |

### `patches/general/`

| Patch | Feature |
|---|---|
| `0001-deterministic-topk.patch` | Deterministic top-k and sampler |
| `0002-deterministic-moe-align.patch` | Deterministic MoE alignment |
| `0003-custom-ar-capacity.patch` | Configurable custom-AR capacity |
| `0004-mnnvl-guard.patch` | MNNVL multicast capability guard |
| `0005-hierarchical-allreduce.patch` | Island-aware hierarchical all-reduce |
| `0006-local-argmax.patch` | Vocab-parallel local argmax infrastructure |
| `0007-multistream-control.patch` | Multi-stream capture safety/control |

### `patches/experimental-sm8x/`

| Patch | Feature |
|---|---|
| `0001-attention-indexer-gemv.patch` | Attention indexer-weights SM80 GEMV |
| `0002-marlin-occupancy.patch` | Marlin occupancy/warp perturbations; fork measured regression |
| `0003-marlin-moe-block-size.patch` | Adaptive Marlin MoE block-size API |
| `0004-mhc-prenorm-cublas.patch` | Restore the fork's T>=32 BF16 cuBLAS prenorm path; selected only by `sm8x-perf` and `all` |
| `0005-sparse-prefill-block-k.patch` | Use BLOCK_K=32 for wide sparse prefill on SM80-class shared memory; selected only by `sm8x-perf` and `all` |

### `patches/experimental-general/`

| Patch | Feature |
|---|---|
| `0001-cudagraph-pad-up.patch` | PIECEWISE graph pad-up using current `max_query_len` API |
| `0002-pinned-staging.patch` | Pinned metadata staging pools and consumers |
| `0003-marlin-moe-workspace.patch` | Persistent Marlin MoE workspace |
| `0004-attention-input-fusion.patch` | Fuse replicated Attention projections |
| `0005-attention-tp-sharding.patch` | TP-shard replicated Attention GEMMs |
| `0006-dspark-vocab-shard.patch` | DSpark vocab-sharded Markov selection |
| `0007-dspark-fused-markov.patch` | DSpark fused sequential Markov kernels |
| `0008-indexer-query-sharding.patch` | Indexer prefill/decode query-row sharding |

### `patches/downstream/`

| Patch | Feature |
|---|---|
| `0001-170hx-serving.patch` | DSpark+PP, long context, parser/API, structured output, aligned O_DIRECT disk KV |

## Remaining fork ports

Ten effective fork optimizations are not yet exported:

- MHC: fused post/sqrsum, prenorm row sharding, fixed split selection, and
  int8 all-reduce ownership (4);
- sparse MLA: LUT decode, 8192-row ragged scan, ratio-128 query blocking,
  uniform decode grouping, decode split tuning, and exact-mask specialization
  (6; the independently ported wide prefill KV tile is no longer part of this
  remaining group).

Two measured-regression knobs (`decode maxnreg`, query-blocked decode) are
recorded separately and are not counted among those ten.

## Build

```bash
cp /path/to/deepseek-v4-cmp170hx/docker/Dockerfile.splitbuild .
cp /path/to/deepseek-v4-cmp170hx/docker/dockerignore.txt .dockerignore
podman build -f Dockerfile.splitbuild -t darknightlab/vllm-170hx:upstream-main .
```

The core patch changes a native header, and some optional patches change CUDA
sources. Use a full sm_80 build for a clean image.

## Verification

Completed:

- all five series apply cleanly from the official pin;
- all five series pass Python compilation and `git diff --check`;
- the `sm8x` series contains only core, SM8x, and downstream paths;
- the `sm8x-perf` series adds only MHC prenorm cuBLAS and sparse-prefill
  BLOCK_K=32 to `sm8x`.

Still required:

- full native build and hardware A/B for the new category series;
- individual hardware certification of experimental patches;
- four-card and sustained long-context testing.
