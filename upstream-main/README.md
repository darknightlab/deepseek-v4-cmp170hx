# Official upstream patch queue

This directory is an experimental rebase of the CMP 170HX stack onto official
`vllm-project/vllm` main. It leaves the existing, hardware-verified `c3046d1`
workflow untouched.

## Pinned base

The queue currently targets:

```text
vllm-project/vllm 402547d7f02bdbfc5dce5d27dc21f50dd4d627b6
2026-08-17 [Bugfix][CI] Release the shared ColBERT engine before test_colbert_hf_comparison (#52608)
```

Main is pinned deliberately. Building from a floating branch makes a two-hour
CUDA build irreproducible and lets upstream changes silently alter kernels. Move
the pin only after rebasing and testing the queue.

## Patch layers

1. `0001-DSv4-SM80-...patch` is the production residual of haosdent's sole
   ahead commit `12810046c799cbe874967e19b1c0fa134ab7b209`, rebased from its
   upstream merge-base `62195e9784ebec1ece42b88a861734e0702cc2d5` onto the
   pinned official main. Newer official implementations are retained, while
   the SM8x backend selection and software FP8 cache encoding are explicitly
   ported. Conflict-coupled query-sharding, MHC, and benchmark experiments are
   excluded; see `REBASE_NOTES.md`.
2. `0002-Apply-deepseek-...patch` ports this repository's DSpark+PP,
   long-context top-k, structured-output, and Responses fixes to the new source
   layout.

The source commit is one commit ahead of its merge-base, but it is not small:
it originally touched 111 files across CUDA/C++, Python/Triton, tests, and
benchmarks. The production residual still changes native sources, so a full
source build is required.

## Prepare a checkout

```bash
./scripts/prepare-upstream-vllm.sh /opt/vllm-upstream
```

The script clones official vLLM when needed, checks out the pinned commit, and
applies both patches with `git am --3way`. It refuses to proceed over local
changes or an unfinished rebase.

To test a newer upstream commit without changing the pin:

```bash
VLLM_REF=origin/main ./scripts/prepare-upstream-vllm.sh /opt/vllm-candidate
```

That is only an applicability probe. A successful three-way apply is not a
correctness result; update `UPSTREAM_VLLM_REF` and regenerate the patch queue
only after source review and hardware tests.

## Build

From the prepared vLLM checkout:

```bash
cp /path/to/deepseek-v4-cmp170hx/docker/Dockerfile.splitbuild .
cp /path/to/deepseek-v4-cmp170hx/docker/dockerignore.txt .dockerignore
podman build -f Dockerfile.splitbuild -t darknightlab/vllm-170hx:upstream-main .
```

This queue modifies `csrc/`, so `VLLM_USE_PRECOMPILED=1` or Python-only bind
mounting is insufficient.

## Verification status

Completed locally:

- confirmed the haosdent branch is exactly one commit ahead of its upstream
  merge-base;
- semantically reviewed all 18 conflict files rather than accepting a clean
  three-way apply as sufficient;
- retained newer official implementations for 16 fork conflicts and ported the
  two required SM80 behaviors into the current source layout;
- removed cross-file-inconsistent query-sharding and MHC TileLang experiment
  remnants from the production candidate;
- ported the downstream patch queue to the current source layout;
- verified the Python tree compiles and both exported patches apply cleanly
  from the pinned commit.

Still required on target hardware:

- full sm_80 native build;
- four-rank PP startup;
- DSpark acceptance/correctness tests;
- long-context and concurrent needle tests;
- disk KV offload startup, bounded-size behavior, eviction, and hit recovery;
- decode/prefill performance comparison against the existing `c3046d1` image.

Do not replace the existing recommended production base until these checks pass.
