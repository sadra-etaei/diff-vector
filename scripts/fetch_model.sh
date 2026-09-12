#!/usr/bin/env bash
# Resumable model fetch.
#
# Why this exists: huggingface_hub reaches for cdn-lfs.huggingface.co, which
# does not resolve on this machine. Plain curl works but DNS here is
# intermittent, and a bare `curl -o` TRUNCATES the partial file on failure --
# we lost 403MB that way. This resumes with -C - and retries until the file
# reaches its known size.

set -u

URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors"
OUT="D:/projects/model diffing/models/Qwen2.5-1.5B-Instruct/model.safetensors"
TARGET=3087467144          # from the HEAD request; the completion test
MAX_ATTEMPTS=60

size_of() { stat -c%s "$OUT" 2>/dev/null || echo 0; }

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    SZ=$(size_of)
    if [ "$SZ" -ge "$TARGET" ]; then
        echo "COMPLETE: $SZ bytes after $((attempt-1)) attempt(s)"
        exit 0
    fi

    PCT=$(( 100 * SZ / TARGET ))
    echo "[attempt $attempt] at ${PCT}% ($SZ / $TARGET) $(date +%H:%M:%S)"

    # -C - resumes from wherever the partial file ended.
    # --retry handles in-transfer blips; the outer loop handles DNS failures,
    # which curl treats as immediately fatal (exit 6) and will not retry.
    curl -sL -C - \
         --retry 10 --retry-delay 5 --retry-all-errors \
         --connect-timeout 30 --max-time 1800 \
         -o "$OUT" "$URL"
    rc=$?
    echo "  curl exit $rc"

    [ "$rc" -eq 0 ] || sleep 10
done

SZ=$(size_of)
if [ "$SZ" -ge "$TARGET" ]; then
    echo "COMPLETE: $SZ bytes"
    exit 0
fi
echo "INCOMPLETE after $MAX_ATTEMPTS attempts: $SZ / $TARGET"
exit 1
