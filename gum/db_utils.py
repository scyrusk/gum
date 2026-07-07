# db_utils.py

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sqlalchemy import (
    MetaData,
    Table,
    select,
    literal_column,
    text,
    func,
)

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import (
    Observation,
    Proposition,
    PropositionFeedback,
    observation_proposition,
)

# Constants
K_DECAY = 2.0      # decay rate for recency adjustment
LAMBDA = 0.5       # trade-off for MMR

def build_fts_query(raw: str, mode: str = "OR") -> str:
    tokens = re.findall(r"\w+", raw.lower())
    if not tokens:
        return ""
    if mode == "PHRASE":
        return f'"{" ".join(tokens)}"'
    elif mode == "OR":
        return " OR ".join(tokens)
    else:  # implicit AND
        return " ".join(tokens)




async def search_propositions_bm25(
    session: AsyncSession,
    user_query: str,
    *,
    limit: int = 3,
    mode: str = "OR",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    include_observations: bool = True,
    enable_decay: bool = True,
    enable_mmr: bool = True,
) -> list[tuple["Proposition", float]]:

    q = build_fts_query(user_query, mode)
    has_query = bool(q)

    # --------------------------------------------------------
    # 1  Build candidate list
    # --------------------------------------------------------
    candidate_pool = limit * 10 if enable_mmr else limit

    if has_query:
        fts_prop = Table("propositions_fts", MetaData())

        if include_observations:
            # --- 1-a-1  WITH observations --------------------
            fts_obs  = Table("observations_fts", MetaData())

            bm25_p   = literal_column("bm25(propositions_fts)").label("score")
            bm25_o   = literal_column("bm25(observations_fts)").label("score")

            sub_p = (
                select(Proposition.id.label("pid"), bm25_p)
                .select_from(
                    fts_prop.join(
                        Proposition,
                        literal_column("propositions_fts.rowid") == Proposition.id,
                    )
                )
                .where(text("propositions_fts MATCH :q"))
            )

            sub_o = (
                select(observation_proposition.c.proposition_id.label("pid"), bm25_o)
                .select_from(
                    fts_obs
                    .join(
                        Observation,
                        literal_column("observations_fts.rowid") == Observation.id,
                    )
                    .join(
                        observation_proposition,
                        observation_proposition.c.observation_id == Observation.id,
                    )
                )
                .where(text("observations_fts MATCH :q"))
            )

            union_sub = sub_p.union_all(sub_o).subquery()

            best_scores = (
                select(
                    union_sub.c.pid,
                    func.min(union_sub.c.score).label("bm25"),
                )
                .group_by(union_sub.c.pid)
                .subquery()
            )
        else:
            # --- 1-a-2  WITHOUT observations -----------------
            best_scores = (
                select(
                    Proposition.id.label("pid"),
                    literal_column("bm25(propositions_fts)").label("bm25"),
                )
                .select_from(
                    fts_prop.join(
                        Proposition,
                        literal_column("propositions_fts.rowid") == Proposition.id,
                    )
                )
                .where(text("propositions_fts MATCH :q"))
                .subquery()
            )

        stmt = (
            select(Proposition, best_scores.c.bm25)
            .join(best_scores, best_scores.c.pid == Proposition.id)
            .order_by(best_scores.c.bm25.asc())          # smallest→best
        )
    else:
        # --- 1-b  No user query ------------------------------
        stmt = (
            select(Proposition, literal_column("0.0").label("bm25"))
            .order_by(Proposition.created_at.desc())
        )

    # --------------------------------------------------------
    # 2  Time filtering & eager-load
    # --------------------------------------------------------
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    if start_time is not None and start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    if start_time is not None:
        stmt = stmt.where(Proposition.created_at >= start_time)
    stmt = stmt.where(Proposition.created_at <= end_time)

    if include_observations:
        stmt = stmt.options(selectinload(Proposition.observations))

    stmt = stmt.limit(candidate_pool)

   # --------------------------------------------------------
    # 3  Execute & score
    # --------------------------------------------------------
    bind = {"q": q} if has_query else {}
    rows = (await session.execute(stmt, bind)).all()
    if not rows:
        return []

    # --- 3-a. Calculate initial scores ---
    initial_scores: list[float] = []
    now = datetime.now(timezone.utc)
    for prop, raw_score in rows:
        relevance_score = -raw_score if has_query else 0.0
        gamma = 0.0
        if enable_decay:
            dt = prop.created_at.replace(tzinfo=timezone.utc)
            age_days = max((now - dt).total_seconds() / 86_400, 0.0)
            alpha = prop.decay if prop.decay is not None else 0.0
            gamma = -alpha * K_DECAY * age_days

        score = relevance_score * math.exp(gamma)
        initial_scores.append(score)

    final_scores_np = np.array(initial_scores)
    min_score = np.min(final_scores_np)
    max_score = np.max(final_scores_np)
    
    if max_score > min_score:
        final_scores_np = (final_scores_np - min_score) / (max_score - min_score)
    else:
        final_scores_np = np.full_like(final_scores_np, 0.5)

    final_scores = final_scores_np.tolist()

    if enable_mmr and len(rows) > 1:
        docs: list[str] = []
        for p, _ in rows:
            doc_parts = [p.text, p.reasoning]
            if include_observations and p.observations:
                obs_concat = " ".join(o.content for o in list(p.observations)[:10])
                doc_parts.append(obs_concat)
            docs.append(" ".join(doc_parts))

        vecs = TfidfVectorizer().fit_transform(docs)
        
        selected_idxs = []
        mmr_scores = np.array(final_scores)

        while len(selected_idxs) < min(limit, len(rows)):
            if not selected_idxs:
                idx = int(np.argmax(mmr_scores))
            else:
                sims = cosine_similarity(vecs, vecs[selected_idxs]).max(axis=1)
                mmr = LAMBDA * mmr_scores - (1 - LAMBDA) * sims
                mmr[selected_idxs] = -np.inf 
                idx = int(np.argmax(mmr))

            selected_idxs.append(idx)
    else:
        idxs = np.argsort(final_scores)[::-1][:limit]
        selected_idxs = idxs.tolist()

    result = [(rows[i][0], final_scores[i]) for i in selected_idxs]    
    return result

async def get_related_observations(
    session: AsyncSession,
    proposition_id: int,
    *,  # Force keyword arguments for optional parameters
    limit: int = 5,
) -> List[Observation]:

    stmt = (
        select(Observation)
        .join(observation_proposition)
        .join(Proposition)
        .where(Proposition.id == proposition_id)
        .order_by(Observation.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def get_recent_propositions(
    session: AsyncSession,
    *,
    limit: int = 10,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    include_observations: bool = False,
) -> List[Proposition]:
    """Fetch the most recent propositions ordered by created_at desc.

    Args:
        session: Active async DB session
        limit: Max number of propositions to return
        start_time: Optional lower bound for created_at
        end_time: Optional upper bound for created_at (defaults to now)
        include_observations: Whether to eager-load related observations

    Returns:
        List[Proposition]: Most recent propositions
    """

    if end_time is None:
        end_time = datetime.now(timezone.utc)
    if start_time is not None and start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    stmt = (
        select(Proposition)
        .where(Proposition.created_at <= end_time)
        .order_by(Proposition.created_at.desc())
        .limit(limit)
    )
    if start_time is not None:
        stmt = stmt.where(Proposition.created_at >= start_time)
    if include_observations:
        stmt = stmt.options(selectinload(Proposition.observations))

    result = await session.execute(stmt)
    return result.scalars().all()


async def get_next_unreviewed_proposition(
    session: AsyncSession,
    *,
    exclude_ids: set[int] | None = None,
) -> "Proposition | None":
    """Most recent proposition (by created_at) that has no feedback yet.

    ``exclude_ids`` additionally skips propositions the user deferred this
    session. Eager-loads observations so the review UI can show the evidence.
    """
    already_reviewed = (
        select(PropositionFeedback.proposition_id)
        .where(PropositionFeedback.proposition_id == Proposition.id)
        .exists()
    )
    stmt = (
        select(Proposition)
        .where(~already_reviewed)
        .order_by(Proposition.created_at.desc())
    )
    if exclude_ids:
        stmt = stmt.where(Proposition.id.notin_(exclude_ids))
    stmt = stmt.options(selectinload(Proposition.observations)).limit(1)
    result = await session.execute(stmt)
    return result.scalars().first()


async def add_proposition_feedback(
    session: AsyncSession,
    proposition: Proposition,
    rating: str,
    note: str | None = None,
) -> PropositionFeedback:
    """Record a rating (accurate/partial/inaccurate) plus optional context note,
    snapshotting the proposition text."""
    fb = PropositionFeedback(
        proposition_id=proposition.id,
        proposition_text=proposition.text,
        reasoning=proposition.reasoning,
        rating=rating,
        note=(note or None),
    )
    session.add(fb)
    await session.flush()
    return fb


async def get_recent_feedback(
    session: AsyncSession, *, limit: int = 8
) -> List[PropositionFeedback]:
    """Most recent judgments, newest first (for few-shot calibration)."""
    result = await session.execute(
        select(PropositionFeedback)
        .order_by(PropositionFeedback.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


def _feedback_relevance(
    feedback: List[PropositionFeedback], query_text: str
) -> list[float]:
    """TF-IDF cosine similarity of each feedback's text vs *query_text*.

    Returns all-zeros when there's no usable query/vocabulary, so selection
    gracefully degrades to recency-only ordering.
    """
    if not query_text or not query_text.strip():
        return [0.0] * len(feedback)
    docs = [f"{fb.proposition_text} {fb.note or ''}" for fb in feedback]
    try:
        vec = TfidfVectorizer().fit(docs)
        sims = cosine_similarity(vec.transform(docs), vec.transform([query_text]))
        return sims.ravel().tolist()
    except ValueError:  # empty vocabulary, etc.
        return [0.0] * len(feedback)


def select_relevant_balanced_feedback(
    feedback: List[PropositionFeedback],
    query_text: str,
    limit: int,
) -> List[PropositionFeedback]:
    """Pick up to *limit* feedback examples that are both relevant to
    *query_text* and balanced across ratings.

    Within each rating class the most relevant examples come first (ties broken
    by recency, since *feedback* is passed newest-first). Slots are then filled
    round-robin across the non-empty classes, so the model always sees
    contrastive signal (accurate vs. partial vs. inaccurate) rather than whatever
    happened to be reviewed most recently. Empty classes yield their slots to
    the others.
    """
    feedback = list(feedback)
    if limit <= 0 or not feedback:
        return []
    if len(feedback) <= limit:
        return feedback

    scores = _feedback_relevance(feedback, query_text)

    # Group by rating, each sorted by relevance desc (stable → recency tiebreak).
    classes: dict[str, list[PropositionFeedback]] = {}
    for fb, score in sorted(
        zip(feedback, scores), key=lambda t: t[1], reverse=True
    ):
        classes.setdefault(fb.rating, []).append(fb)

    order = [r for r in ("accurate", "partial", "inaccurate") if classes.get(r)]
    order += [r for r in classes if r not in order]  # any unexpected ratings

    chosen: List[PropositionFeedback] = []
    cursor = {r: 0 for r in order}
    while len(chosen) < limit and any(cursor[r] < len(classes[r]) for r in order):
        for r in order:
            if len(chosen) >= limit:
                break
            if cursor[r] < len(classes[r]):
                chosen.append(classes[r][cursor[r]])
                cursor[r] += 1
    return chosen


async def count_review_progress(session: AsyncSession) -> tuple[int, int]:
    """Return (total_propositions, reviewed_count)."""
    total = (await session.execute(select(func.count(Proposition.id)))).scalar() or 0
    reviewed = (
        await session.execute(
            select(func.count(func.distinct(PropositionFeedback.proposition_id))).where(
                PropositionFeedback.proposition_id.isnot(None)
            )
        )
    ).scalar() or 0
    return total, reviewed


async def get_recent_observations(
    session: AsyncSession,
    *,
    limit: int = 10,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> List[Observation]:
    """Fetch the most recent observations ordered by created_at desc."""
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    if start_time is not None and start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    stmt = (
        select(Observation)
        .where(Observation.created_at <= end_time)
        .order_by(Observation.created_at.desc())
        .limit(limit)
    )
    if start_time is not None:
        stmt = stmt.where(Observation.created_at >= start_time)

    result = await session.execute(stmt)
    return result.scalars().all()