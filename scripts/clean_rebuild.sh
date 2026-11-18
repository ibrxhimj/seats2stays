#!/usr/bin/env bash
# Phase 5, Step 5.2 — clean rebuild. Rebuilds the whole project from the raw files in a fresh folder with a fresh Python
# environment, then compares every number and every table with the 5.1 snapshot.
#
#   scripts/clean_rebuild.sh [DEST] [PYTHON]        default DEST=../atp2026_rebuild, PYTHON=python3.12
#
# Copied: code, tests, web page, config, requirements, the raw competition files and the reference/external inputs.
# Not copied: anything the pipeline makes (data/processed, data/interim, reports, build) — those are rebuilt.
# Needs internet once, for pip. Takes about as long as running every phase (see the timings it prints).
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$(realpath -m "${1:-$SRC/../atp2026_rebuild}")"
PY="${2:-python3.12}"
[ "$DEST" != "$SRC" ] || { echo "DEST must not be the project itself"; exit 2; }
[ -f "$SRC/data/processed/phase5/snapshot.json" ] || { echo "no snapshot: run 'make snapshot' first"; exit 2; }

step() { echo; echo "== $1  ($(date +%H:%M:%S))"; }
step "copy the sources to $DEST"
rm -rf "$DEST"; mkdir -p "$DEST/data" "$DEST/reports"
cp -r "$SRC"/{src,tests,web,scripts,config.yaml,Makefile,requirements.txt,requirements-app.txt,Dockerfile,.dockerignore,docker-compose.yml,README.md} "$DEST"/
[ -d "$SRC/docs" ] && cp -r "$SRC/docs" "$DEST"/
cp -r "$SRC"/data/{raw,reference,external} "$DEST/data"/
find "$DEST" -name __pycache__ -prune -exec rm -rf {} +

step "fresh environment: $PY + requirements.txt (exact pins)"
"$PY" -m venv "$DEST/.venv"
"$DEST/.venv/bin/pip" install -q --upgrade pip
"$DEST/.venv/bin/pip" install -q -r "$DEST/requirements.txt"
"$DEST/.venv/bin/python" --version

cd "$DEST"
export PYTHONPATH=src PY="$DEST/.venv/bin/python"
unset ATP_ROOT
for t in data phase1 phase2 phase3 phase4data sobol ahead benchmark audit limits; do      # Tab 3 reads phase5/sobol, ahead and limitations (benchmark and audit feed the list)
  step "make $t"; SECONDS=0; make -s "$t" PY="$PY"; echo "   ($SECONDS s)"
done

step "snapshot of the rebuild"
"$PY" -m atp.phase5 snapshot

step "compare with the project's snapshot"
cd "$SRC"
"$DEST/.venv/bin/python" -m atp.phase5 compare "$DEST"
