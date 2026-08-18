#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATCH_DIR="$PROJECT_ROOT/upstream-main/patches"
PINNED_REF="$(tr -d '[:space:]' < "$PROJECT_ROOT/upstream-main/UPSTREAM_VLLM_REF")"
DEST="${1:-$PROJECT_ROOT/.work/vllm-upstream}"
VLLM_REF="${VLLM_REF:-$PINNED_REF}"
VLLM_FORK_PERFORMANCE_PROFILE="${VLLM_FORK_PERFORMANCE_PROFILE:-sm8x}"

if [[ ! -d "$DEST/.git" ]]; then
    mkdir -p "$(dirname "$DEST")"
    git clone https://github.com/vllm-project/vllm.git "$DEST"
else
    ORIGIN_URL="$(git -C "$DEST" remote get-url origin 2>/dev/null || true)"
    if [[ "$ORIGIN_URL" != "https://github.com/vllm-project/vllm" &&
          "$ORIGIN_URL" != "https://github.com/vllm-project/vllm.git" &&
          "$ORIGIN_URL" != "git@github.com:vllm-project/vllm.git" ]]; then
        echo "ERROR: $DEST origin is not vllm-project/vllm: $ORIGIN_URL" >&2
        exit 1
    fi
fi

if [[ -n "$(git -C "$DEST" status --porcelain)" ]]; then
    echo "ERROR: $DEST has uncommitted changes; refusing to overwrite them." >&2
    exit 1
fi

if [[ -d "$DEST/.git/rebase-apply" || -d "$DEST/.git/rebase-merge" ]]; then
    echo "ERROR: $DEST has an unfinished am/rebase operation." >&2
    exit 1
fi

git -C "$DEST" fetch origin main
git -C "$DEST" checkout --detach "$VLLM_REF"

if [[ "$VLLM_REF" != "$PINNED_REF" ]]; then
    echo "WARNING: patches were rebased and verified against $PINNED_REF, not $VLLM_REF." >&2
fi

SERIES_FILE="$PATCH_DIR/series/$VLLM_FORK_PERFORMANCE_PROFILE.txt"
if [[ ! -f "$SERIES_FILE" ]]; then
    echo "ERROR: unknown performance profile: $VLLM_FORK_PERFORMANCE_PROFILE" >&2
    echo "Expected one of: none, sm8x, verified, all." >&2
    exit 1
fi

PATCHES=()
while IFS= read -r relative_path; do
    [[ -z "$relative_path" || "$relative_path" == \#* ]] && continue
    patch="$PATCH_DIR/$relative_path"
    if [[ ! -f "$patch" ]]; then
        echo "ERROR: series entry not found: $patch" >&2
        exit 1
    fi
    PATCHES+=("$patch")
done < "$SERIES_FILE"

if [[ ${#PATCHES[@]} -eq 0 ]]; then
    echo "ERROR: empty patch series: $SERIES_FILE" >&2
    exit 1
fi

git -C "$DEST" \
    -c user.name="deepseek-v4-cmp170hx patcher" \
    -c user.email="noreply@localhost" \
    am --3way "${PATCHES[@]}"

printf 'Prepared vLLM checkout at %s\n' "$DEST"
printf 'Base: %s\n' "$VLLM_REF"
printf 'Performance profile: %s\n' "$VLLM_FORK_PERFORMANCE_PROFILE"
printf 'Result: %s\n' "$(git -C "$DEST" rev-parse HEAD)"
