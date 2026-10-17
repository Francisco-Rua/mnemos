#!/usr/bin/env bash
# Build the Apple Foundation Model bridge (desktop/fm-helper/main.swift) into src-tauri/resources/fm-helper.
# Needs the macOS 26 SDK (Xcode 26). Without it, or with --placeholder, writes a stub that reports "not available",
# so the app always bundles the file and the hub simply never offers Apple Intelligence.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
out="$here/src-tauri/resources/fm-helper"
mkdir -p "$(dirname "$out")"
if [ "${1:-}" != "--placeholder" ] && [ -n "${FM_HELPER_BIN:-}" ] && [ -f "$FM_HELPER_BIN" ]; then
  cp "$FM_HELPER_BIN" "$out" && chmod +x "$out" && echo "fm-helper: using $FM_HELPER_BIN" && exit 0
fi
if [ "${1:-}" != "--placeholder" ] && command -v swiftc >/dev/null \
   && swiftc -O -parse-as-library -target arm64-apple-macos26.0 "$here/fm-helper/main.swift" -o "$out" 2>/tmp/fm-helper.log; then
  chmod +x "$out" && echo "fm-helper: built for macOS 26 (arm64)" && exit 0
fi
cat /tmp/fm-helper.log 2>/dev/null | tail -5 || true
if [ -n "${FM_HELPER_REQUIRE:-}" ]; then echo "fm-helper: build failed" >&2; exit 1; fi
printf '#!/bin/sh\necho %s\n' "'{\"available\": false, \"reason\": \"built without the macOS 26 SDK\"}'" > "$out"
chmod +x "$out"
echo "fm-helper: placeholder (no macOS 26 SDK here)"
