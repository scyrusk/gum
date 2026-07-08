# test_memory_curation.py
#
# Stdlib-only (unittest) tests for the concurrency contract between Memory-page
# curation (paper Fig 3B: delete_proposition / update_proposition) and the
# background batch processor.  Runnable without pytest or a live model:
#     python -m unittest tests.test_memory_curation
#
# The bug these guard against: a batch (_process_batch) loads existing
# propositions with their observations collections and revises them across slow
# LLM calls inside one transaction. If the user forgets one of those
# propositions from the Memory page during that window, the batch's writes used
# to reference rows that had vanished underneath them and died with
# sqlalchemy StaleDataError:
#     DELETE ... on 'observation_proposition' expected to delete N row(s);
#     Only M were matched.
# Curation stays lock-free (a batch is inference-bound and can run for minutes,
# so blocking deletes on it makes them appear broken); instead the batch's own
# writes tolerate a proposition disappearing beneath them.

from __future__ import annotations

import tempfile
import unittest
import uuid

from gum import gum as Gum
from gum.models import Observation, Proposition, observation_proposition
from gum.schemas import PropositionItem
from sqlalchemy import insert, select


def _prop(text: str, confidence: int) -> Proposition:
    return Proposition(
        text=text,
        reasoning=f"because of {text}",
        confidence=confidence,
        decay=5,
        revision_group=uuid.uuid4().hex,
        version=1,
    )


class _Base(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.gum = Gum("Omar", "dummy-model", data_directory=self._tmp.name, db_name="test.db")
        await self.gum.connect_db()

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    async def _seed_prop_with_obs(self, text: str, n_obs: int) -> int:
        """Persist a proposition backed by n_obs observations; return its id."""
        async with self.gum._session() as s:
            prop = _prop(text, 8)
            obs = [
                Observation(observer_name="screen", content=f"{text} obs {i}", content_type="input_text")
                for i in range(n_obs)
            ]
            s.add_all([prop, *obs])
            await s.flush()
            await s.execute(
                insert(observation_proposition),
                [{"observation_id": o.id, "proposition_id": prop.id} for o in obs],
            )
            return prop.id

    async def _count_junction(self) -> int:
        async with self.gum._session() as s:
            return len((await s.execute(select(observation_proposition))).all())


class CurationCrudTests(_Base):
    async def test_delete_removes_prop_and_cascades_junction(self):
        pid = await self._seed_prop_with_obs("wedding in Chicago", 2)
        self.assertEqual(await self._count_junction(), 2)
        self.assertTrue(await self.gum.delete_proposition(pid))
        self.assertEqual(await self._count_junction(), 0)
        async with self.gum._session() as s:
            self.assertIsNone(await s.get(Proposition, pid))

    async def test_delete_missing_returns_false(self):
        self.assertFalse(await self.gum.delete_proposition(999_999))

    async def test_update_edits_fields(self):
        pid = await self._seed_prop_with_obs("needs formal wear", 1)
        prop = await self.gum.update_proposition(pid, text="rented a tuxedo", confidence=9)
        self.assertIsNotNone(prop)
        self.assertEqual(prop.text, "rented a tuxedo")
        self.assertEqual(prop.confidence, 9)

    async def test_update_missing_returns_none(self):
        self.assertIsNone(await self.gum.update_proposition(999_999, text="x"))

    async def test_curation_does_not_hold_batch_lock(self):
        # Regression: curation must stay responsive and never wedge the batch loop.
        pid = await self._seed_prop_with_obs("stays responsive", 1)
        await self.gum.delete_proposition(pid)
        self.assertFalse(self.gum._batch_processing_lock.locked())


class BatchToleratesConcurrentDeleteTests(_Base):
    """The batch's DB writes must survive a proposition being forgotten mid-batch."""

    async def test_handle_similar_survives_forget_during_revise(self):
        # Two propositions the batch has decided are SIMILAR and will revise.
        p1 = await self._seed_prop_with_obs("Omar is going to a wedding", 3)
        p2 = await self._seed_prop_with_obs("Omar attends a formal event", 2)

        # A fresh batch observation, and a stub revise that simulates the user
        # clicking "Forget" on p1 (via its own committed session) partway through
        # the slow revise call — exactly the race that produced StaleDataError.
        async def fake_revise(related_obs, similar_cluster):
            await self.gum.delete_proposition(p1)
            return [
                PropositionItem(
                    reasoning="merged reasoning",
                    proposition="Omar is attending a formal wedding",
                    confidence=8,
                    decay=5,
                )
            ]

        self.gum._revise_propositions = fake_revise  # type: ignore[assignment]

        async with self.gum._session() as session:
            new_obs = Observation(
                observer_name="screen", content="RSVP yes", content_type="input_text"
            )
            session.add(new_obs)
            await session.flush()
            similar = [
                await session.get(Proposition, p1),
                await session.get(Proposition, p2),
            ]
            # Load the observations collections so the ORM has the pre-delete state
            # cached — this is what used to go stale.
            for p in similar:
                _ = await self.gum_related(session, p.id)
            # Must NOT raise StaleDataError.
            await self.gum._handle_similar(session, similar, [new_obs])

        # Both old propositions are gone (p1 forgotten by the user, p2 replaced by
        # the revision) and exactly one revised proposition now exists. We compare
        # by text, not id: SQLite reuses the freed rowids for the new proposition.
        async with self.gum._session() as s:
            texts = [
                p.text for p in (await s.execute(select(Proposition))).scalars().all()
            ]
            self.assertEqual(texts, ["Omar is attending a formal wedding"])
            self.assertNotIn("Omar is going to a wedding", texts)
            self.assertNotIn("Omar attends a formal event", texts)

    async def test_attach_obs_skips_forgotten_proposition(self):
        # The identical/different path attaches new observations to an existing
        # proposition. If that proposition was forgotten concurrently, the attach
        # must no-op instead of raising a foreign-key IntegrityError.
        pid = await self._seed_prop_with_obs("existing belief", 1)
        async with self.gum._session() as session:
            prop = await session.get(Proposition, pid)
            new_obs = Observation(
                observer_name="screen", content="more evidence", content_type="input_text"
            )
            session.add(new_obs)
            await session.flush()
            # Simulate the concurrent Memory-page delete landing now.
            await self.gum.delete_proposition(pid)
            # Must NOT raise even though prop.id no longer exists.
            await self.gum._attach_obs_if_missing(prop, new_obs, session)

        # No orphaned junction row was created for the deleted proposition.
        self.assertEqual(await self._count_junction(), 0)

    @staticmethod
    async def gum_related(session, prop_id):
        from gum.db_utils import get_related_observations

        return await get_related_observations(session, prop_id)


if __name__ == "__main__":
    unittest.main()
