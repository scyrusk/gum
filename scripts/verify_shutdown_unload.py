"""Verify the GUM unloads local models when stopped.

Standalone, no daemon / live model needed: we monkeypatch
``llm._native_generate_ping`` to record every (url, model, keep_alive) ping,
then assert that shutdown fires a keep_alive=0 (immediate-unload) ping for both
the vision and text targets. Mirrors the isolation pattern used to verify the
keep-warm pinger (iter9).

Run: python scripts/verify_shutdown_unload.py
"""

import asyncio

from gum import llm


async def main() -> None:
    pings: list[tuple[str, str, object]] = []

    def fake_ping(url: str, model: str, keep_alive) -> None:
        pings.append((url, model, keep_alive))

    llm._native_generate_ping = fake_ping  # type: ignore[assignment]

    # Two Ollama /v1 targets sharing one base (vision + text).
    targets = [
        ("http://localhost:11434/v1", "qwen2.5vl:7b"),
        ("http://localhost:11434/v1", "qwen2.5:32b"),
    ]

    await llm.release_models(targets)

    assert pings, "release_models sent no pings"
    assert all(ka == 0 for _, _, ka in pings), f"expected keep_alive=0, got {pings}"
    models = {m for _, m, _ in pings}
    assert models == {"qwen2.5vl:7b", "qwen2.5:32b"}, f"unexpected models: {models}"
    # Both resolve to the same native /api/generate url.
    assert all(url.endswith("/api/generate") for url, _, _ in pings), pings
    print(f"OK: shutdown released {len(models)} model(s) with keep_alive=0: {sorted(models)}")

    # A base that isn't an Ollama /v1 endpoint resolves to nothing (no unloads).
    pings.clear()
    await llm.release_models([("https://api.example.com/openai", "gpt-4o")])
    assert not pings, f"non-/v1 target should be skipped, got {pings}"
    print("OK: non-/v1 endpoint skipped on shutdown")

    # Integration: drive the real gum.stop_update_loop over a minimal stub and
    # confirm it releases BOTH the vision and text targets on shutdown.
    import sys
    from types import SimpleNamespace

    import gum.gum  # noqa: F401  (register the submodule in sys.modules)

    gum_mod = sys.modules["gum.gum"]  # the module, not the re-exported class
    GumClass = gum_mod.gum

    class _VisionObs:
        warm_targets = [("http://localhost:11434/v1", "qwen2.5vl:7b")]

    released: list[list[tuple[str, str]]] = []

    async def fake_release(tgts, *, logger=None):
        released.append(list(tgts))

    gum_mod.release_models = fake_release  # type: ignore[assignment]

    stub = SimpleNamespace(
        observers=[_VisionObs()],
        _api_base="http://localhost:11434/v1",
        model="qwen2.5:32b",
        logger=llm.logging.getLogger("gum.verify"),
        _loop_task=None,
        _batch_task=None,
        _warm_task=None,
        batcher=None,
    )
    stub._vision_warm_targets = lambda: GumClass._vision_warm_targets(stub)
    stub._all_warm_targets = lambda: GumClass._all_warm_targets(stub)
    await GumClass.stop_update_loop(stub)

    assert released, "stop_update_loop did not release any models"
    got = set(released[0])
    assert got == {
        ("http://localhost:11434/v1", "qwen2.5vl:7b"),
        ("http://localhost:11434/v1", "qwen2.5:32b"),
    }, f"stop_update_loop released wrong targets: {got}"
    print(f"OK: stop_update_loop released vision + text targets: {sorted(got)}")


if __name__ == "__main__":
    asyncio.run(main())
