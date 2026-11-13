#!/usr/bin/env bash
# Generic resumable Hugging Face file fetch (see fetch_model.sh for why curl).
# Usage: fetch_hf.sh URL OUT TARGET_BYTES
# Exit: 0 complete, 1 incomplete after all attempts, 2 file larger than TARGET.

set -u

if [ "$#" -ne 3 ]; then
    echo "usage: $0 URL OUT TARGET_BYTES" >&2
    exit 64
fi

URL="$1"
OUT="$2"
TARGET="$3"
MAX_ATTEMPTS=60

mkdir -p "$(dirname "$OUT")"

size_of() { stat -c%s "$OUT" 2>/dev/null || echo 0; }

check_oversize() {
    local sz
    sz=$(size_of)
    if [ "$sz" -gt "$TARGET" ]; then
        echo "ERROR: $OUT is $sz bytes, LARGER than expected $TARGET." >&2
        echo "       The remote file may have changed; a resumed download may be corrupt. Delete and re-fetch." >&2
        exit 2
    fi
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    check_oversize
    SZ=$(size_of)
    if [ "$SZ" -ge "$TARGET" ]; then
        echo "COMPLETE: $SZ bytes after $((attempt-1)) attempt(s)"
        exit 0
    fi

    if [ "$TARGET" -gt 0 ]; then PCT=$(( 100 * SZ / TARGET )); else PCT=0; fi
    echo "[attempt $attempt] at ${PCT}% ($SZ / $TARGET) $(date +%H:%M:%S)"

    curl -sL -C - \
         --retry 10 --retry-delay 5 --retry-all-errors \
         --connect-timeout 30 --max-time 1800 \
         -o "$OUT" "$URL"
    rc=$?
    echo "  curl exit $rc"

    [ "$rc" -eq 0 ] || sleep 10
done

check_oversize
SZ=$(size_of)
if [ "$SZ" -ge "$TARGET" ]; then
    echo "COMPLETE: $SZ bytes"
    exit 0
fi
echo "INCOMPLETE after $MAX_ATTEMPTS attempts: $SZ / $TARGET"
exit 1
