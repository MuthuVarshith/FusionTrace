#!/usr/bin/env bash
# Assemble the contents of a Hugging Face Space (Docker SDK) from a commit.
#
#   deploy/hf-spaces/prepare_space.sh [OUT_DIR] [REV]
#
# OUT_DIR defaults to .space-build/ at the repository root (gitignored) and must
# not exist or be empty. REV defaults to HEAD. Nothing is uploaded: push OUT_DIR
# yourself as described in docs/deploy-hf-spaces.md.
#
# Only committed files are used (git archive), so local data, uploads, model
# weights and .env files can never leak into the Space. Notebooks, tests and the
# sample media under backend/test/ are left out: a Space repository must never
# carry real user media, and the app does not need them.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
out="${1:-$repo_root/.space-build}"
rev="${2:-HEAD}"

if [ -e "$out" ] && [ -n "$(ls -A "$out")" ]; then
    echo "prepare_space: $out exists and is not empty; remove it or pick another directory" >&2
    exit 1
fi
mkdir -p "$out"

git -C "$repo_root" archive --format=tar "$rev" \
    Dockerfile .dockerignore requirements.txt LICENSE deploy/docker backend \
    | tar -x -C "$out"
rm -rf "$out/backend/notebook" "$out/backend/test" "$out/backend/tests"

# The Space card (front matter: sdk docker, app_port 7860) becomes the README.
git -C "$repo_root" show "$rev:deploy/hf-spaces/README.md" > "$out/README.md"

# Space-specific defaults. Space variables set in the UI still override these.
cat >> "$out/Dockerfile" <<'EOF'

# ── Hugging Face Spaces (appended by deploy/hf-spaces/prepare_space.sh) ──
# Spaces route traffic to app_port 7860 through their reverse proxy, so the
# proxy's X-Forwarded-For header is trusted for the client address.
ENV PORT=7860 \
    FORWARDED_ALLOW_IPS=*
EXPOSE 7860
EOF

echo "Space contents ready in $out (from $(git -C "$repo_root" rev-parse --short "$rev"))."
echo "Review it, then upload, e.g.: hf upload <user>/<space> \"$out\" . --repo-type space"
