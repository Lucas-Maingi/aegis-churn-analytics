#!/usr/bin/env bash
# Deploy the current main tree to the Hugging Face Space.
#
# HF Spaces require: a docker README with app metadata at the repo root, a
# Space-specific Dockerfile (port 7860, seeding), and binary files tracked via
# Git LFS. This builds a throwaway orphan snapshot with those swaps and
# force-pushes it to the Space's main branch, leaving your working branch
# untouched.
#
# Usage: bash deploy/push_space.sh   (run from the repo root, on a clean tree)
set -euo pipefail

REMOTE="${SPACE_REMOTE:-space}"
SNAPSHOT="space-snapshot-$$"
START_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is dirty — commit or stash first." >&2
  exit 1
fi

cleanup() {
  git checkout -q "$START_BRANCH"
  git branch -D "$SNAPSHOT" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git checkout -q --orphan "$SNAPSHOT"

# Swap in the Space-specific entrypoints
cp deploy/Dockerfile.hf Dockerfile
cp deploy/README.hf.md README.md

# LFS-track model artifacts (HF rejects raw binaries)
printf '*.joblib filter=lfs diff=lfs merge=lfs -text\n' > .gitattributes
git lfs install --local >/dev/null

# The orphan checkout inherits main's index with the joblibs as plain blobs;
# clearing and re-adding forces the LFS clean filter to run over them.
git rm -r --cached --quiet .
git add -A
git commit -q -m "deploy: Hugging Face Space snapshot (multi-tenant SaaS + feedback loop)"

# Sanity-check the artifacts really are LFS pointers before pushing
if ! git lfs ls-files | grep -q 'model.joblib'; then
  echo "model.joblib was not LFS-tracked — aborting." >&2
  exit 1
fi
git push "$REMOTE" "$SNAPSHOT:main" --force

echo "Pushed snapshot to '$REMOTE' main."
