#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then echo "Run ./install.sh first."; exit 1; fi
echo "Starting HUVEC Damage Analyzer at http://localhost:8501 (Ctrl+C to stop)"
exec .venv/bin/python -m streamlit run app.py
