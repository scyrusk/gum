TRANSCRIPTION_PROMPT = """Transcribe in markdown ALL the content from the screenshots of the user's screen.

NEVER SUMMARIZE ANYTHING. You must transcribe everything EXACTLY, word for word, but don't repeat yourself.

ALWAYS include all the application names, file paths, and website URLs in your transcript.

Create a FINAL structured markdown transcription."""

SUMMARY_PROMPT = """Provide a detailed description of the actions occuring across the provided images. The images are in the order they were taken.

Include as much relevant detail as possible, but remain concise.

Generate a handful of bullet points and reference *specific* actions the user is taking.

Keep in mind that that the content on the screen is what the user is viewing. It may not be what the user is actively doing or what they believe, so practice caution when making assumptions."""

# Single-call prompt that produces BOTH the word-for-word transcription and the
# action summary in one vision request. Collapsing the two per-observation
# vision calls into one roughly halves local VLM latency per observation — the
# most frequent inference the GUM performs — without dropping either signal the
# downstream proposition model consumes. The images are passed in chronological
# order with the CURRENT (last) screenshot last; transcription is scoped to that
# current screen (the earlier frames are near-duplicates a few seconds apart, so
# re-transcribing them adds output tokens without new information), while the
# action summary still spans every frame for temporal context.
COMBINED_PROMPT = """You are analyzing a chronological sequence of screenshots of the user's screen. The LAST image is the CURRENT state of the screen; any earlier images are recent prior states, in order.

Respond with EXACTLY two markdown sections, in this order:

## Transcription
Transcribe in markdown ALL the content visible in the CURRENT (last) screenshot. NEVER SUMMARIZE ANYTHING here — transcribe everything EXACTLY, word for word, but don't repeat yourself. ALWAYS include all application names, file paths, and website URLs.

## Summary
Provide a detailed but concise description of the actions occurring across the images, in the order they were taken. Generate a handful of bullet points that reference *specific* actions the user is taking. Keep in mind that the content on the screen is what the user is viewing — it may not be what the user is actively doing or what they believe, so practice caution when making assumptions."""