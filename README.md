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
gum observations --date 7/7/2026  # all observations from a given Eastern-time day
gum observations --date 7/7/2026 -o day.txt   # write results to a file (exports full content)

gum agenda                      # ranked radar of your open commitments & deadlines
gum agenda --window 7           # only commitments due within 7 days (overdue/undated always shown)
gum agenda --json               # machine-readable JSON (add -s / --sanitize to pseudonymize PII)

gum review                      # open a browser GUI to judge propositions True/False (see below)
```

`gum start` accepts overrides, e.g. `gum start --vision-model qwen2.5vl:32b --text-model gpt-oss:20b --port 8500`. Logs stream to `~/.cache/gum/gum.log`.

> **Note on models & memory.** On first `gum start`, the GUM automatically creates lean, context-capped copies of your models (`gum-<model>-ctx<N>`) via a tiny Modelfile. Ollama otherwise loads models at their full 128K context, whose KV cache balloons a 7B vision model to ~50 GB and forces the vision and text models to constantly evict/reload each other — which shows up as a very hot machine and propositions that never appear. The capped copies (defaults: vision 16K → ~14 GB, text 32K → ~30 GB) stay resident together. Tune with `GUM_VISION_NUM_CTX` / `GUM_TEXT_NUM_CTX`.
>
> **Idle behaviour.** The vision model is kept warm continuously (it runs on nearly every interaction). The larger text model runs only on batches, so it is released from memory after ~10 min with no new observations (`GUM_TEXT_IDLE_UNLOAD`, seconds) and reloaded on the next batch — a cooler, lighter idle in exchange for a one-time reload. Set `GUM_TEXT_IDLE_UNLOAD=0` to keep it always resident.

#### Blacklist proposition content

To prevent the text model from generating propositions about sensitive topics, create `~/.cache/gum/blacklist.txt` with one natural-language rule per line:

```text
# Lines beginning with # are comments.
Do not generate propositions related to passwords or authentication secrets.
Do not generate propositions related to credit cards or bank account details.
Do not generate propositions containing adult or explicit content.
```

Blank lines and comments are ignored. The file is read before every proposition-generation and revision call, so changes apply to the next batch without restarting `gum`. When all possible propositions would violate a rule, the model is instructed to generate none. The blacklist controls newly generated and revised propositions; it does not remove propositions already stored in `gum.db`.

Set `GUM_BLACKLIST_FILE` in `.env` if you want to keep the file elsewhere. If the configured file does not exist, blacklist filtering is disabled.

### 5. Review propositions to improve the model

`gum review` opens a small browser GUI that shows you one proposition at a time — along with the observations that led to it — and asks whether it's **Accurate**, **Somewhat** accurate, or **Inaccurate** about you. You can add an optional free-text **note** with any rating to give the model context (e.g. "only when I'm coding, not for writing"). Keyboard: `t` / `s` / `f` for the three ratings, `k` to skip. It walks your most recent unreviewed propositions first.

```bash
gum review          # opens http://127.0.0.1:8423 in your browser
```

Your judgments (and notes) are stored locally and fed back into the proposition generator as few-shot calibration examples, so the model learns the kinds of inferences that are accurate about you, refines toward your notes on partially-right ones, and avoids the inaccurate ones. This runs fine alongside the background daemon (nothing leaves your machine).

For each batch the examples are chosen to be **relevant** to the current activity (TF-IDF/cosine over your judged propositions) and **balanced across ratings** (round-robin over accurate/partial/inaccurate), so the model always gets contrastive, on-topic signal rather than just the most recent judgments. Tune with `GUM_FEWSHOT_LIMIT` (examples per batch, default 8; `0` disables) and `GUM_FEWSHOT_POOL` (recent candidates considered, default 200).

### 6. Menu-bar app (macOS)

Prefer clicking to typing? A small menu-bar companion lets you start/stop the GUM, search it, and browse the most recent propositions without leaving the toolbar.

```bash
pip install "gum-ai[tray]"      # one-time: adds rumps
gum tray                        # 🧠 appears in the menu bar (💤 when stopped)
```

From the menu you can **Start / Stop GUM**, **Search GUM…** (results in a pop-up), skim **Recent Propositions** (click one for its full reasoning and confidence), **Open Review UI**, or **Open Logs**. It drives the same daemon and localhost API as the CLI, so it works alongside everything above.

#### Launch it at login

Because `gum tray` is a CLI command (not an `.app`), the cleanest way to start it at login is a **LaunchAgent**. Create `~/Library/LaunchAgents/ai.gum.tray.plist` — replace `/path/to/gum` with your checkout and `/path/to/gum/.venv/bin/gum` with the `gum` on your `PATH` (`which gum`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.gum.tray</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/gum/.venv/bin/gum</string>
        <string>tray</string>
    </array>
    <!-- So the tray finds your .env (USER_NAME, etc.) as it does in the terminal. -->
    <key>WorkingDirectory</key>
    <string>/path/to/gum</string>
    <key>RunAtLoad</key>
    <true/>
    <!-- false so "Quit GUM" from the menu sticks until the next login. -->
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>/tmp/gum-tray.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/gum-tray.log</string>
</dict>
</plist>
```

Then manage it with `launchctl`:

```bash
# Enable + start now (registers it to launch at every login):
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.gum.tray.plist

# Stop it now and disable auto-start at login:
launchctl bootout gui/$(id -u)/ai.gum.tray

# Remove it permanently:
launchctl bootout gui/$(id -u)/ai.gum.tray   # if still loaded
rm ~/Library/LaunchAgents/ai.gum.tray.plist
```

The tray auto-starts, but it won't begin observing until you click **Start GUM** (or run `gum start`). Note that a LaunchAgent won't appear in **System Settings → Login Items** — that list only shows `.app` bundles.

### 7. Build on it from other apps

While the GUM is running it serves a **localhost-only REST API** (default `http://127.0.0.1:8422`) that any local app, in any language, can query:

```bash
curl "http://127.0.0.1:8422/query?q=email&limit=5"
curl "http://127.0.0.1:8422/recent?limit=5"
curl "http://127.0.0.1:8422/observations?limit=5"
curl "http://127.0.0.1:8422/agenda?limit=5&window_days=14"   # ranked commitment radar
```

The `/agenda` endpoint returns the same ranked commitment/deadline radar as the
`gum agenda` CLI and the MCP `agenda` tool, as JSON; like every response it is
pseudonymized when the server runs with `--sanitize`.

Or use it from Python directly (`from gum import gum; await gum(...).query("email")`), or wire up an [MCP server](docs/tutorials/mcp.md) for Claude Desktop / Codex — including the built-in, PII-sanitized `gum mcp` (fail-closed and on by default) that lets a local agent pull your context on demand and hand the finished artifact back through `gum rehydrate`.

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
