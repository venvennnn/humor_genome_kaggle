# AGENTS.md

## Cursor Cloud specific instructions

This repo is a single Python/Streamlit app ("Why'd They Laugh?"). Standard setup, run,
and test commands live in `README.md` (Quickstart, CLI, Tests). Prefer those; notes below
only cover non-obvious caveats for this environment.

### Services
- **Streamlit web app** (`app.py`) — the product UI, served on port 8501. This is the one
  service that must run.
- No database, queue, or other backend services exist.

### Backend selection caveat (important)
- The Gemma reasoning core is pluggable (`ollama` / `hf` / `mock`), auto-selected in that
  order. In the Cloud VM there is **no Ollama server and no GPU**, so run with the offline
  `mock` backend explicitly to avoid slow auto-detection/HF downloads:
  `HUMOR_GENOME_BACKEND=mock`. The mock backend is deterministic and exercises the full UI
  and CLI. Real Gemma output requires running Ollama (`ollama run gemma3`) separately.

### Running / testing (non-obvious bits)
- Pip installs to the user site; console scripts land in `~/.local/bin`, which is not on
  PATH. Invoke tools as modules instead: `python3 -m streamlit run app.py` and
  `python3 -m pytest`.
- Run the app headless in the Cloud VM:
  `HUMOR_GENOME_BACKEND=mock python3 -m streamlit run app.py --server.port 8501 --server.headless true --server.address 0.0.0.0`
  Health check: `curl http://localhost:8501/_stcore/health` returns `ok`.
- Tests (`pytest tests/`) run entirely against the mock backend — no external services or
  ffmpeg needed.
- `ffmpeg`/`ffprobe` are only needed for the video feature (frame/audio extraction) and are
  already available in this environment.
