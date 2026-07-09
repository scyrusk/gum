# Using MCPs to connect to GUMs

## I just want to set up the MCP

First, you'll need to set up the GUM in general and have it build some sense of your context. To do this, follow the instructions on [the front page here](../index.md). You'll also need a client that supports MCP. One example client is the MacOS Claude Desktop app, which you can download [here](https://claude.ai/download). The Claude desktop app requires the uv package manager for MCP, so you'll need to follow [the instructions on the uv website](https://docs.astral.sh/uv/getting-started/installation/) (or simply ```brew install uv```).

!!! note "If you didn't use brew, make sure uv is installed _globally_"
    Annoyingly, some apps (Claude) don't look at your local PATH. So if you didn't use brew, your uv might be in your local bin ```~/.local/bin/uv```. You can test this by running ```which uv```. Luckily, you can just set a symlink to fix this:

    ```
    sudo ln -s ~/.local/bin/uv /usr/local/bin/uv
    ```

### Option 1: One-click Desktop Extension (DXT)

DXTs are [extension files](https://github.com/anthropics/dxt) that make the MCP setup really easy. First, make sure the Claude desktop app is updated! Download the .dxt file from [the releases page here](https://github.com/GeneralUserModels/gumcp/releases) and just double-click (or drag it into the extensions page in the Claude Desktop app; Claude > Settings > Extensions). You'll be asked to provide your full name so the GUM knows who you are. Don't forget to enable the extension, and you'll be good to go!

### Option 2: Manual Setup

Clone the [MCP Repository](https://github.com/GeneralUserModels/gumcp) and run the following:

```bash
> git clone git@github.com:GeneralUserModels/gumcp.git
> cd gumcp
```

In the gumcp folder, create a .env file with your environment variables. All you need is a user name in the file (e.g.```USER_NAME="Omar Shaikh"```). In sum, the contents of your .env file look something like this:

```bash
USER_NAME="Omar Shaikh"
```

Finally, install the MCP client, pointing to the .env file:

```bash
> uv run mcp install server.py -f .env --with gum-ai
```

The MCP should then be enabled in the Claude app!

!!! note "The MCP **only** connects clients like the Claude app to the GUM."
    Simply enabling the MCP does not mean the GUM is learning. You still need to have the background GUM process running (`gum start`) to build the underlying database of propositions (e.g. from the instructions on [the front page here.](../index.md))

!!! tip "Running fully local"
    The MCP reads the same local SQLite model (`~/.cache/gum/gum.db`) keyed by `USER_NAME`, so set the **same `USER_NAME`** you gave `gum start`. With the Ollama setup, both the GUM and the MCP operate entirely on-device. (If you prefer a plain HTTP interface over MCP, the running GUM also serves a localhost REST API at `http://127.0.0.1:8422`.)

### Option 3: Built-in `gum mcp` (sanitized by default)

The GUM ships its own MCP server so a local executing agent (Claude Desktop, Codex, …) can pull **sanitized** context on demand — for example, ask your agent to *"draft a grant proposal for the Schmidt Foundation"* and it will call the `gather_context` tool to acquire the relevant propositions from your model before writing, without you pasting anything.

Unlike the external `gumcp` above, this server pseudonymizes PII **on egress and fail-closed** (it will not start if the sanitizer can't load), so raw identities never leave the machine even when the agent relays context to a frontier model.

Point your MCP client at the command directly (this is the JSON block Claude Desktop / Codex use under the hood):

```json
{
  "mcpServers": {
    "gum-context": {
      "command": "gum",
      "args": ["mcp", "--user-name", "Omar Shaikh"]
    }
  }
}
```

It exposes two tools: `gather_context(topic, limit)` (relevance-ranked propositions for a task) and `recent_context(limit)` (a snapshot of recent activity). Sanitization requires the extra (`pip install 'gum-ai[sanitize]'`); for a fully-local, trusted agent you can serve raw propositions with `gum mcp --no-sanitize`.

## Sanitizing output for off-device / frontier models

If you want to feed GUM's observations and propositions to a model that runs **off your machine** (e.g. a frontier model behind the MCP), you can have GUM pseudonymize PII on the way out. Detected entities (names, emails, phone numbers, addresses, etc.) are replaced with **consistent pseudo-IDs** — the same real person always reads as `[PERSON_1]`, so the downstream model can still reason that "an email to person X" and "a follow-up text to person X" concern the same person, without ever seeing the real identity.

First install the extra (it pulls a small local PII-detection model):

```bash
pip install 'gum-ai[sanitize]'
```

**CLI export** (safe to paste into any external model):

```bash
gum observations -d 7/7/2026 -s -o day.md   # -s / --sanitize
gum query -s "email drafts"
gum recent -s
```

**Sanitized REST API** — start the daemon with `--sanitize` so *every* API response is pseudonymized. This is fail-closed: if the sanitizer can't load, the API refuses to start rather than serving raw data.

```bash
gum start --sanitize        # or set GUM_SANITIZE=1
curl http://127.0.0.1:8422/health   # -> {"sanitized": true, ...}
```

!!! warning "The external `gumcp` reads the database directly, bypassing sanitization"
    The external `gumcp` (Options 1–2) reads `~/.cache/gum/gum.db` **directly** — so `gum start --sanitize` (which sanitizes the *REST API*, not the raw DB) does **not** automatically sanitize what *that* MCP returns. To get sanitized data to an off-device model with it, either (a) use the CLI `-s` export path above, or (b) point your MCP client at the sanitized **REST API** (`http://127.0.0.1:8422`). The built-in **`gum mcp`** server (Option 3) sidesteps this entirely: it reads the DB directly *and* pseudonymizes every proposition on the way out, fail-closed and on by default.

The entity ↔ pseudo-ID map (the re-identification key) is stored separately in `~/.cache/gum/entities.db`, isolated from the main model so it can be locked down or excluded from any export. Tune detection with `GUM_SANITIZE_MIN_SCORE` and pick the model with `GUM_SANITIZE_MODEL` (see `.env.example`).

## Try it out!

Here's an example of what happens when I prompt Claude and it uses the MCP:

<div style="text-align: center;">
<img src="../mcp.png" alt="Figure 1" width="60%" />
</div>

## Tutorial

(coming soon: a walkthrough on how this was built!)
