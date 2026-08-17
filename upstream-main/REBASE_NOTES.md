# Rebase notes

## Provenance

```text
haosdent head:       12810046c799cbe874967e19b1c0fa134ab7b209
haosdent merge-base: 62195e9784ebec1ece42b88a861734e0702cc2d5
official base:       402547d7f02bdbfc5dce5d27dc21f50dd4d627b6
```

`git rev-list --count <merge-base>..<haosdent-head>` returns `1`.

A direct cherry-pick onto the official base automatically applied 93 files and
reported 18 conflicts. An initial mechanical resolution kept the official-main
version of every conflicted file. That queue applied cleanly, but semantic
review found two source-level inconsistencies in the automatically applied
files:

- `sparse_attn_indexer.py` expected query-sharding fields and helpers that the
  official `mla/indexer.py` metadata does not provide;
- the fork's MHC TileLang wrapper called three kernels absent from the official
  `tilelang_kernels.py` selected during conflict resolution.

Those two experimental feature groups were restored to the official-main
implementation as a unit. Their conflict-coupled tests and the fork's benchmark
artifacts are not carried in this production candidate. Three focused SM80
kernel tests remain in the patch. The required constructor half of the SM80
indexer fallback is retained separately: CUDA devices without DeepGEMM now
pre-warm and use the Triton MQA kernels, without importing any query-sharding
metadata. The first-layer MHC broadcast path likewise selects the existing
TileLang prenorm GEMM when DeepGEMM is unavailable, matching the fallback
already used by the other MHC pre paths.

The full build also exposed two native compile errors. The fork's persistent
top-k rewrite indexed two histogram buffers with an out-of-scope `iter`
identifier; both sites now use the function's `radix_iter` parameter, matching
its existing global-round calculation. The pinned official base also declared
`fused_gdn_decode_post_conv_mtp()` under the KDA feature guard while registering
it under the GDN guard; the declaration now uses `VLLM_ENABLE_FUSED_GDN_DECODE`,
which is required by the SM80 build configuration.

The semantic review retained official main for 16 conflict files and ported the
required fork behavior into two conflict files:

- `vllm/models/deepseek_v4/nvidia/model.py`: select the Ampere Triton sparse-MLA
  backend on SM8x and reject the Blackwell-only FP4 indexer cache;
- `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`: use the
  software E4M3 encoder for all three FP8 cache-write sites on SM80, and retain
  the measured compressor warp selection.

The other conflict files use the newer official implementations in the fork
layer. `model_runner.py` receives the repository's separate DSpark+PP additions
in the second patch:

```text
tests/kernels/test_compressor_kv_cache.py
tests/models/test_dspark_mla.py
tests/v1/worker/test_kv_block_zeroer.py
vllm/model_executor/kernels/mhc/tilelang_kernels.py
vllm/model_executor/layers/fused_moe/experts/marlin_moe.py
vllm/model_executor/models/qwen3_dspark.py
vllm/models/deepseek_v4/amd/rocm.py
vllm/models/deepseek_v4/attention.py
vllm/models/deepseek_v4/nvidia/dspark.py
vllm/parser/engine/streaming_parser_engine.py
vllm/v1/attention/backends/mla/indexer.py
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/gpu/spec_decode/dspark/speculator.py
vllm/v1/worker/gpu/structured_outputs.py
vllm/v1/worker/utils.py
```

Relevant later upstream work includes KV block-zeroing fixes (#50276, #51749,
#52058), DSpark scheduling (#47808), DSV4 plain/MTP/DSpark correctness (#51538),
KV layout changes (#51704), and structured-output fixes (#52436). Choosing the
old fork side wholesale for these conflicts would discard those changes.

Fork-only conflict hunks not ported are optional or superseded paths, including
adaptive Marlin workspace/block sizing, MHC fused-sqrsum and int8 all-reduce
call sites, indexer query/decode sharding, and DSpark vocab-sharded Markov
fusion. In particular, the DSpark fork hunks predate the official confidence
head and adaptive-verification implementation, so replacing those files would
regress current DSpark correctness. They can be reconsidered as isolated
follow-up patches with dedicated tests and A100 measurements.

After the first layer, the repository's existing patches were replayed. Patches
0003, 0005, 0005a, 008, and 009 applied directly. Patch 0006 was re-expressed
against the official indexer flow: it row-chunks `fp8_fp4_mqa_logits` without
requiring fork query-sharding metadata. Patches 0002 and 0004 were manually
relocated to the current `SpeculativeConfig` and GPU model-runner structure
without changing their DSpark+PP behavior. Patch 0001 remains omitted because
its guard was already upstream. The disk backend now stores arbitrary PP KV
payload sizes in 4096-byte-aligned slot rows: GPU DMA copies only the real
payload, while O_DIRECT reads and writes the padded stride and capacity is
bounded using that physical stride.

This is a source-level rebase, not a hardware certification. The SM80 backend
selection and software FP8 encoding are now source-consistent, but only a full
sm_80 build and four-GPU run can validate kernel compilation, correctness, and
performance.
