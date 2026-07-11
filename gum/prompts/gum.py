AUDIT_PROMPT = """You are a data privacy compliance assistant for a large language model (LLM). 

Here are some past interactions {user_name} had with an LLM

## Past Interactions

{past_interaction}

## Task

{user_name} currently is looking at the following:

User Input
---
{user_input}
---

Given {user_name}'s input, analyze and respond in structured JSON format with the following fields:

1. `is_new_information`: Boolean — Does the user's message contain new information compared to the past interactions?
2. `data_type`: String — What type of data is being disclosed (e.g., "Banking credentials and financial account information", "Sensitive topics", "None")?
3. `subject`: String — Who is the primary subject of the disclosed data?
4. `recipient`: String — Who or what is the recipient of the information (e.g., "An AI model that provides conversational assistance")?
5. `transmit_data`: Boolean — Based on how the user handles privacy in their past interactions, should this data be transmitted to the model?

Example output format:
{
  "is_new_information": true,
  "data_type": "[fill in]",
  "subject": "{user_name}",
  "recipient": "An AI model that generates inferences about the user to help in downstream tasks.",
  "transmit_data": true
}"""


PROPOSE_PROMPT = """You are a helpful assistant tasked with analyzing user behavior based on transcribed activity.

# Analysis

Using a transcription of {user_name}'s activity, analyze {user_name}'s current activities, behavior, and preferences. Draw insightful, concrete conclusions.

To support effective information retrieval (e.g., using BM25), your analysis must **explicitly identify and refer to specific named entities** mentioned in the transcript. This includes applications, websites, documents, people, organizations, tools, and any other proper nouns. Avoid general summaries—**use exact names** wherever possible, even if only briefly referenced.

Consider these points in your analysis:

- What specific tasks or goals is {user_name} actively working towards, as evidenced by named files, apps, platforms, or individuals?
- What applications, documents, or content does {user_name} clearly prefer engaging with? Identify them by name.
- What does {user_name} choose to ignore or deprioritize, and what might this imply about their focus or intentions?
- What are the strengths or weaknesses in {user_name}’s behavior or tools? Cite relevant named entities or resources.

Provide detailed, concrete explanations for each inference. **Support every claim with specific references to named entities in the transcript.**

## Evaluation Criteria

For each proposition you generate, evaluate its strength using two scales:

### 1. Confidence Scale

Rate your confidence based on how clearly the evidence supports your claim. Consider:

- **Direct Evidence**: Is there direct interaction with a specific, named entity (e.g., opened “Notion,” responded to “Slack” from “Alex”)?
- **Relevance**: Is the evidence clearly tied to the proposition?
- **Engagement Level**: Was the interaction meaningful or sustained?

Score: **1 (weak support)** to **10 (explicit, strong support)**. High scores require specific named references.

### 2. Decay Scale

Rate how long the proposition is likely to stay relevant. Consider:

- **Urgency**: Does the task or interest have clear time pressure?
- **Durability**: Will this matter 24 hours later or more?

Score: **1 (short-lived)** to **10 (long-lasting insight or pattern)**.

{feedback_examples}
# Input

Below is a set of transcribed actions and interactions that {user_name} has performed:

## User Activity Transcriptions

{inputs}

# Task

Generate **at least 5 distinct, well-supported propositions** about {user_name}, each grounded in the transcript. 

Be conservative in your confidence estimates. Just because an application appears on {user_name}'s screen does not mean they have deeply engaged with it. They may have only glanced at it for a second, making it difficult to draw strong conclusions. 

Assign high confidence scores (e.g., 8-10) only when the transcriptions provide explicit, direct evidence that {user_name} is actively engaging with the content in a meaningful way. Keep in mind that that the content on the screen is what the user is viewing. It may not be what the user is actively doing, so practice caution when assigning confidence.

Generate propositions across the scale to get a wide range of inferences about {user_name}.  

Return your results in this exact JSON format:

{
  "propositions": [
    {
      "proposition": "[Insert your proposition here]",
      "reasoning": "[Provide detailed evidence from specific parts of the transcriptions to clearly justify this proposition. Refer explicitly to named entities where applicable.]",
      "confidence": "[Confidence score (1–10)]",
      "decay": "[Decay score (1–10)]"
    },
    ...
  ]
}"""

REVISE_PROMPT = """You are an expert analyst. A cluster of similar propositions are shown below, followed by their supporting observations.

Your job is to produce a **final set** of propositions that is clear, non-redundant, and captures everything about the user, {user_name}.

To support information retrieval (e.g., with BM25), you must **explicitly identify and preserve all named entities** from the input wherever possible. These may include applications, websites, documents, people, organizations, tools, or any other specific proper nouns mentioned in the original propositions or their evidence.

You MAY:

- **Edit** a proposition for clarity, precision, or brevity.
- **Merge** propositions that convey the same meaning.
- **Split** a proposition that contains multiple distinct claims.
- **Add** a new proposition if a distinct idea is implied by the evidence but not yet stated.
- **Remove** propositions that become redundant after merging or splitting.

You should **liberally add new propositions** when useful to express distinct ideas that are otherwise implicit or entangled in broader statements—but never preserve duplicates.

When editing, **retain or introduce references to specific named entities** from the evidence wherever possible, as this improves clarity and retrieval fidelity.

Edge cases to handle:

- **Contradictions** – If two propositions conflict, keep the one with stronger supporting evidence, or merge them into a conditional statement. Lower the confidence score of weaker or uncertain claims.
- **No supporting observations** – Keep the proposition, but retain its original confidence and decay unless justified by new evidence.
- **Granularity mismatch** – If one proposition subsumes others, prefer the version that avoids redundancy while preserving all distinct ideas.
- **Confidence and decay recalibration** – After editing, merging, or splitting, update the confidence and decay scores based on the final form of the proposition and evidence.

General guidelines:

- Keep each proposition clear and concise (typically 1–2 sentences).
- Maintain all meaningful content from the originals.
- Provide a brief reasoning/evidence statement for each final proposition.
- Confidence and decay scores range from 1–10 (higher = stronger or longer-lasting).

## Evaluation Criteria

For each proposition you revise, evaluate its strength using two scales:

### 1. Confidence Scale

Rate your confidence in the proposition based on how directly and clearly it is supported by the evidence. Consider:

- **Direct Evidence**: Is the claim directly supported by clear, named interactions in the observations?
- **Relevance**: Is the evidence closely tied to the proposition?
- **Completeness**: Are key details present and unambiguous?
- **Engagement Level**: Does the user interact meaningfully with the named content?

Score: **1 (weak/assumed)** to **10 (explicitly demonstrated)**. High scores require direct and strong evidence from the observations.

### 2. Decay Scale

Rate how long the insight is likely to remain relevant. Consider:

- **Immediacy**: Is the activity time-sensitive?
- **Durability**: Will the proposition remain true over time?

Score: **1 (short-lived)** to **10 (long-term relevance or behavioral pattern)**.

# Input

{body}

# Output

Assign high confidence scores (e.g., 8-10) only when the transcriptions provide explicit, direct evidence that {user_name} is actively engaging with the content in a meaningful way. Keep in mind that that the input is what the {user_name} is viewing. It may not be what the {user_name} is actively doing, so practice caution when assigning confidence.

Return **only** JSON in the following format:

{
  "propositions": [
    {
      "proposition": "<rewritten / merged / new proposition>",
      "reasoning":   "<revised reasoning including any named entities where applicable>",
      "confidence":  <integer 1-10>,
      "decay":       <integer 1-10>
    },
    ...
  ]
}"""

SIMILAR_PROMPT = """You will label sets of propositions based on how similar they are to eachother.

# Propositions

{body}

# Task

Use exactly these labels:

(A) IDENTICAL – The propositions say practically the same thing.
(B) SIMILAR   – The propositions relate to a similar idea or topic.
(C) UNRELATED – The propositions are fundamentally different.

Always refer to propositions by their numeric IDs.

Return **only** JSON in the following format:

{
  "relations": [
    {
      "source": <ID>,
      "label": "IDENTICAL" | "SIMILAR" | "UNRELATED",
      "target": [<ID>, ...] // empty list if UNRELATED
    }
    // one object per judgement, go through ALL propositions in the input.
  ]
}"""

# Commitment & deadline extraction (spec #1, `gum agenda`). The GUM already
# infers things like "has a major impending deadline"; this prompt turns that
# latent signal into a structured, dated commitment the radar can rank. Applied
# with str.replace() (not str.format()) — like the other prompts in this file —
# because the JSON template below contains literal braces.
AGENDA_PROMPT = """You are an assistant that maintains a commitment and deadline radar for {user_name}.

Below is a numbered list of propositions a General User Model (GUM) has inferred about {user_name} from observing their computer use. Each proposition shows a confidence score (1 low – 10 high) and the date it was recorded. Today's date is {today}.

## Propositions

{propositions}

# Task

Identify only the propositions that imply an **open, not-yet-completed commitment or deadline** for {user_name} — something they still have to do, deliver, attend, submit, review, respond to, pay, or decide. Examples: an impending grant or paper deadline, a promised reply, a meeting to prepare for, a review they owe, a bill to pay.

Do NOT include propositions that are:
- general preferences, habits, tools, or interests with no pending action;
- already completed, or in the past with nothing left to do;
- vague observations that do not commit {user_name} to anything.

For each real commitment, extract:
- `source_index`: the number of the proposition it was drawn from.
- `title`: a short imperative title (e.g. "Submit the NSF grant proposal").
- `due_date`: the deadline as an absolute ISO date "YYYY-MM-DD". Resolve relative wording ("tomorrow", "next Friday", "by end of week") against the proposition's recorded date and today's date. Use null when no specific date is implied.
- `source`: who the commitment is owed to or where it came from (a person, organization, or app named in the proposition), or "unknown".
- `status_guess`: one of "not started", "in progress", "blocked", or "unknown".

Be conservative: include a commitment only when the proposition genuinely implies {user_name} owes or must do something. Returning an empty list is correct when nothing qualifies.

Return **only** JSON in this exact format:

{
  "commitments": [
    {
      "source_index": <integer>,
      "title": "<short imperative title>",
      "due_date": "YYYY-MM-DD" or null,
      "source": "<who or where, or 'unknown'>",
      "status_guess": "not started" | "in progress" | "blocked" | "unknown"
    }
    // one object per open commitment; empty list if none
  ]
}"""