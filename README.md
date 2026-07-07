# GUM (General User Models)

[![arXiv](https://img.shields.io/badge/arXiv-2505.10831-b31b1b.svg)](https://arxiv.org/abs/2505.10831)

General User Models learn about you by observing any interaction you have with your computer. The GUM takes as input any unstructured observation of a user (e.g., device screenshots) and constructs confidence-weighted propositions that capture the user's knowledge and preferences. GUMs introduce an architecture that infers new propositions about a user from multimodal observations, retrieves related propositions for context, and continuously revises existing propositions.

## Fully local (Ollama)

This fork runs the entire pipeline on your own machine through [Ollama](https://ollama.com) — **no screenshots or propositions leave your computer by default** (a privacy guard refuses non-local inference endpoints unless you set `GUM_ALLOW_REMOTE=1`).

### 1. Install Ollama and pull the models

```bash
# Install Ollama from https://ollama.com, then pull one vision + one text model
ollama pull qwen2.5vl:7b        # vision: transcribes screenshots (~6 GB)
ollama pull qwen2.5-coder:32b   # text: writes propositions with reliable JSON (~19 GB)

# Keep both models resident so they don't reload between roles
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_KEEP_ALIVE=30m
ollama serve                    # usually already running as a background service
```

### 2. Install the GUM

```bash
git clone <this-repo> && cd gum
python3 -m venv .venv && source .venv/bin/activate
pip install --editable .

cp .env.example .env            # then edit .env and set USER_NAME to your full name
```

### 3. Grant macOS permissions (one-time)

The GUM needs to watch your screen and input. In **System Settings → Privacy & Security**, enable your terminal app (e.g. Terminal or iTerm) under both:
- **Accessibility** (to detect mouse/keyboard activity)
- **Screen Recording** (to capture screenshots)

You may need to restart the terminal after granting these.

### 4. Run it

```bash
gum start                       # begin observing in the background + serve the local API
gum status                      # check it's running and see the latest proposition
gum stop                        # stop whenever you like

# Inspect what it has learned
gum recent                      # list the most recent propositions
gum query "email" -l 10         # BM25 search over propositions
gum observations                # list the most recent raw observations (--full for complete text)
```

`gum start` accepts overrides, e.g. `gum start --vision-model qwen2.5vl:32b --text-model gpt-oss:20b --port 8500`. Logs stream to `~/.cache/gum/gum.log`.

> **Note on models & memory.** On first `gum start`, the GUM automatically creates lean, context-capped copies of your models (`gum-<model>-ctx<N>`) via a tiny Modelfile. Ollama otherwise loads models at their full 128K context, whose KV cache balloons a 7B vision model to ~50 GB and forces the vision and text models to constantly evict/reload each other — which shows up as a very hot machine and propositions that never appear. The capped copies (defaults: vision 16K → ~14 GB, text 32K → ~30 GB) stay resident together. Tune with `GUM_VISION_NUM_CTX` / `GUM_TEXT_NUM_CTX`.

### 5. Build on it from other apps

While the GUM is running it serves a **localhost-only REST API** (default `http://127.0.0.1:8422`) that any local app, in any language, can query:

```bash
curl "http://127.0.0.1:8422/query?q=email&limit=5"
curl "http://127.0.0.1:8422/recent?limit=5"
curl "http://127.0.0.1:8422/observations?limit=5"
```

Or use it from Python directly (`from gum import gum; await gum(...).query("email")`), or wire up the [MCP server](docs/tutorials/mcp.md) for Claude Desktop.

### How often does it observe?

Observation is **activity-driven, not on a fixed timer** — an idle machine generates nothing.

- **Screenshots → vision model:** the GUM continuously buffers frames at ~10 fps in memory, but only saves a before/after screenshot pair and sends it to the vision model after a burst of mouse activity settles (a **2-second** debounce of no movement/clicks/scrolls). So roughly one vision pass per interaction burst, a couple seconds after you pause. Nothing is captured while you're idle, and you can skip specific apps.
- **Propositions → text model:** transcribed observations are batched. The text model runs the propose/revise step once a batch fills up — every **5 observations** by default (tunable with `--min-batch-size` / `--max-batch-size`, up to 15). So propositions update roughly every few interaction bursts, in the background.

The 10 fps / 2-second-debounce capture cadence lives in `gum/observers/screen.py` (`_CAPTURE_FPS`, `_DEBOUNCE_SEC`); batch sizes are exposed on `gum start`.

## Documentation

**Please go here for documentation on setting up and using GUMs: [https://generalusermodels.github.io/gum/docs/](https://generalusermodels.github.io/gum/docs/)**

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Citation and Paper

If you're interested in reading more, please check out our paper!

[Creating General User Models from Computer Use](https://arxiv.org/abs/2505.10831)

```bibtex
@misc{shaikh2025creatinggeneralusermodels,
    title={Creating General User Models from Computer Use}, 
    author={Omar Shaikh and Shardul Sapkota and Shan Rizvi and Eric Horvitz and Joon Sung Park and Diyi Yang and Michael S. Bernstein},
    year={2025},
    eprint={2505.10831},
    archivePrefix={arXiv},
    primaryClass={cs.HC},
    url={https://arxiv.org/abs/2505.10831}, 
}
```
