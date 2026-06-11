#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh  –  One-shot setup for QuPath YOLO Active Learning extension
# Run from the repo root: bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

YOLO_AL_DIR="$HOME/.qupath/yolo-al"
VENV_DIR="$YOLO_AL_DIR/yolo-venv"

echo "=== QuPath YOLO-AL Setup ==="

# 1. Create yolo-al directory inside QuPath user data
mkdir -p "$YOLO_AL_DIR"

# 2. Copy Python server files
echo "[1/4] Installing Python server → $YOLO_AL_DIR"
cp python/yolo_server.py   "$YOLO_AL_DIR/"
cp python/requirements.txt "$YOLO_AL_DIR/"

# 3. Create virtual environment and install dependencies
echo "[2/4] Creating Python venv at $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "[3/4] Installing Python packages (this may take a few minutes)…"
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$YOLO_AL_DIR/requirements.txt" -q

echo "[4/4] Building QuPath extension JAR…"
./gradlew jar

EXTENSIONS_DIR="$HOME/.qupath/extensions"
mkdir -p "$EXTENSIONS_DIR"
cp build/libs/qupath-yolo-al-*.jar "$EXTENSIONS_DIR/"
echo "      JAR copied to: $EXTENSIONS_DIR"

echo ""
echo "✓ Setup complete."
echo ""
echo "Next steps:"
echo "  1. Start QuPath 0.6"
echo "  2. Extensions → YOLO Active Learning → Open Panel…"
echo "  3. Set dataset directory, annotate slides, then Extract → Train → Infer"
