# schemas.py

from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

class AuditSchema(BaseModel):
    """
    Output produced by the privacy-audit LLM call.
    """
    is_new_information: bool = Field(..., description="Whether the message reveals anything not seen before")
    data_type:          str  = Field(..., description="Category of data being disclosed")
    subject:            str  = Field(..., description="Who the data is about")
    recipient:          str  = Field(..., description="Who receives the data")
    transmit_data:      bool = Field(..., description="Should downstream processing continue")

    model_config = ConfigDict(extra="forbid")

class PropositionItem(BaseModel):
    reasoning: str = Field(..., description="The reasoning for the proposition")
    proposition: str = Field(..., description="The proposition string")
    confidence: Optional[int] = Field(
        ...,
        description="Confidence score from 1 (low) to 10 (high)"
    )
    decay: Optional[int] = Field(
        ...,
        description="Decay score from 1 (low) to 10 (high)"
    )

    model_config = ConfigDict(extra="forbid")

class PropositionSchema(BaseModel):
    propositions: List[PropositionItem] = Field(
        ...,
        description="Up to K propositions"
    )
    model_config = ConfigDict(extra="forbid")

class BlacklistComplianceSchema(BaseModel):
    allowed_indices: List[int] = Field(
        ...,
        description="Zero-based indices of propositions that comply with every blacklist rule",
    )

    model_config = ConfigDict(extra="forbid")

class Update(BaseModel):
    content: str = Field(..., description="The content of the update")
    content_type: Literal["input_text", "input_image"] = Field(..., description="The type of the update")

RelationLabel = Literal["IDENTICAL", "SIMILAR", "UNRELATED"]

class RelationItem(BaseModel):
    source: int                     = Field(description="Proposition ID")
    label:  RelationLabel           = Field(description="Relationship label")

    # give target a default_factory so the JSON‐schema default is [] (allowed)
    target: List[int] = Field(
        default_factory=list,
        description="IDs of other propositions (empty if none)"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["source", "label", "target"]
        }
    )


class RelationSchema(BaseModel):
    relations: List[RelationItem]

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["relations"]
        }
    )

class SuggestionItem(BaseModel):
    """A single proactive suggestion GUMBO derives from the user's GUM.

    The four score fields feed the mixed-initiative expected-utility decision
    (Horvitz [36]) that GUMBO uses to decide whether a suggestion is worth
    surfacing — see gum.gumbo.expected_utility. All scores are on a 1 (low) to
    10 (high) scale so the local text model has a consistent, easy target, and
    that range is enforced (ge=1/le=10) rather than merely documented: these
    scores are gate inputs to the execution bridge — ``probability_useful`` is
    read raw by Executor.is_auto_dispatchable and the rest feed the
    ``should_surface`` decision it also depends on — so a malformed out-of-range
    value must be rejected (driving structured_completion's retries) instead of
    sailing through the auto-dispatch gate, mirroring RiskAssessmentSchema.risk.
    """
    title: str = Field(..., description="Short imperative title, e.g. 'Draft the wedding-travel budget'")
    description: str = Field(..., description="What GUMBO proposes to do or has already drafted on the user's behalf")
    rationale: str = Field(..., description="Why this is relevant right now, grounded in the provided propositions")
    probability_useful: int = Field(
        ..., ge=1, le=10, description="How likely the user finds any value in this suggestion, P(useful), from 1 (low) to 10 (high)"
    )
    benefit: int = Field(
        ..., ge=1, le=10, description="Benefit to the user if the suggestion is completed, from 1 (low) to 10 (high)"
    )
    cost_if_wrong: int = Field(
        ..., ge=1, le=10, description="Cost of interrupting the user with this if it is not useful (false positive), from 1 to 10"
    )
    cost_if_missed: int = Field(
        ..., ge=1, le=10, description="Cost to the user of never seeing this if it is useful (false negative), from 1 to 10"
    )

    model_config = ConfigDict(extra="forbid")


class SuggestionSchema(BaseModel):
    suggestions: List[SuggestionItem] = Field(
        ..., description="Candidate suggestions for the user, most relevant first"
    )
    model_config = ConfigDict(extra="forbid")


class CommitmentItem(BaseModel):
    """A single open commitment/deadline the agenda extractor pulled from one
    proposition.

    ``source_index`` ties the commitment back to the numbered proposition it was
    extracted from, so the ranker can reuse that proposition's confidence, decay,
    and age (see gum.agenda) instead of asking the model to re-derive them.
    """
    source_index: int = Field(
        ..., description="1-based index of the proposition this commitment was extracted from"
    )
    title: str = Field(
        ..., description="Short imperative title of the commitment, e.g. 'Submit the NSF grant proposal'"
    )
    due_date: Optional[str] = Field(
        ...,
        description="Absolute due date as an ISO 8601 'YYYY-MM-DD' string, or null if the proposition implies no specific date",
    )
    source: str = Field(
        ...,
        description="Who the commitment is owed to or where it came from (a named person, org, or app), or 'unknown'",
    )
    status_guess: str = Field(
        ...,
        description="Best guess at status: 'not started', 'in progress', 'blocked', or 'unknown'",
    )

    model_config = ConfigDict(extra="forbid")


class CommitmentSchema(BaseModel):
    commitments: List[CommitmentItem] = Field(
        ..., description="Open commitments/deadlines found in the propositions; empty if none qualify"
    )
    model_config = ConfigDict(extra="forbid")


class CommitmentVerdictSchema(BaseModel):
    """Verdict from the agenda's second-pass verification of one candidate.

    The extraction pass classifies commitments while looking at the *whole*
    candidate pool, which pressures the model to promote borderline ongoing
    activities to fill the list. This verdict re-judges a single candidate in
    isolation — a cleaner binary decision — so those false positives can be
    dropped before ranking (see gum.agenda.CommitmentRadar).
    """
    is_commitment: bool = Field(
        ..., description="True only if the proposition implies a genuine discrete, dischargeable commitment"
    )
    reason: str = Field(
        ..., description="One short phrase justifying the verdict"
    )

    model_config = ConfigDict(extra="forbid")


ReversibilityClass = Literal["read_only", "reversible", "irreversible"]


class RiskAssessmentSchema(BaseModel):
    """Safety classification of the action a GUMBO suggestion would require.

    Produced by the execution bridge's risk-assessment LLM call (see
    gum.executor.Executor.assess_risk). The executor uses ``reversibility`` and
    ``risk`` to gate auto-dispatch: only read-only/reversible, low-risk actions on
    a high-confidence suggestion may run automatically; everything else stays
    proposal-only, held for the user's explicit approval.
    """

    reversibility: ReversibilityClass = Field(
        ...,
        description=(
            "How undoable the action is: 'read_only' (only reads/searches/drafts, "
            "changes nothing relied upon), 'reversible' (changes something trivially "
            "undone, e.g. a local draft file), or 'irreversible' (effects that cannot "
            "be cleanly undone or that reach outside the machine, e.g. sending a "
            "message, a purchase, deleting data)."
        ),
    )
    risk: int = Field(
        ...,
        ge=1,
        le=10,
        description=(
            "How much harm a wrong or unwanted action could cause the user, from 1 "
            "(trivial, easily ignored) to 10 (severe, hard to recover from)."
        ),
    )
    rationale: str = Field(..., description="One sentence justifying the classification")

    model_config = ConfigDict(extra="forbid")


def get_schema(json_schema):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "json_output",
            "schema": json_schema,
        },
    }

UPDATE_MAP = {
    "input_text": "text",
    "input_image": "image_url",
}
