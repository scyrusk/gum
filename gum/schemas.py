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
    10 (high) scale so the local text model has a consistent, easy target.
    """
    title: str = Field(..., description="Short imperative title, e.g. 'Draft the wedding-travel budget'")
    description: str = Field(..., description="What GUMBO proposes to do or has already drafted on the user's behalf")
    rationale: str = Field(..., description="Why this is relevant right now, grounded in the provided propositions")
    probability_useful: int = Field(
        ..., description="How likely the user finds any value in this suggestion, P(useful), from 1 (low) to 10 (high)"
    )
    benefit: int = Field(
        ..., description="Benefit to the user if the suggestion is completed, from 1 (low) to 10 (high)"
    )
    cost_if_wrong: int = Field(
        ..., description="Cost of interrupting the user with this if it is not useful (false positive), from 1 to 10"
    )
    cost_if_missed: int = Field(
        ..., description="Cost to the user of never seeing this if it is useful (false negative), from 1 to 10"
    )

    model_config = ConfigDict(extra="forbid")


class SuggestionSchema(BaseModel):
    suggestions: List[SuggestionItem] = Field(
        ..., description="Candidate suggestions for the user, most relevant first"
    )
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
