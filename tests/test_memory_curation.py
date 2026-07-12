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
from unittest import mock

from gum import gum as Gum
from gum.models import (
    AgendaItem,
    AgendaOverride,
    Observation,
    Proposition,
    observation_proposition,
)
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


class AgendaOverrideMethodTests(_Base):
    """Agenda edits persist an override and propagate into the GUM, staying
    lock-free like the other curation paths (the override table is independent of
    the batch, so an edit never blocks on an in-flight batch)."""

    async def _one_override(self) -> AgendaOverride | None:
        async with self.gum._session() as s:
            return (await s.execute(select(AgendaOverride))).scalar_one_or_none()

    async def test_edit_rewrites_single_date_and_records_override(self):
        pid = await self._seed_prop_with_obs("Submit the grant by 2026-07-20", 1)
        with mock.patch.object(self.gum.batcher, "push") as push:
            ok = await self.gum.apply_agenda_override(pid, due_date="2026-08-15")
        self.assertTrue(ok)
        ov = await self._one_override()
        self.assertIsNotNone(ov)
        self.assertEqual(ov.due_date, "2026-08-15")
        async with self.gum._session() as s:
            text = (await s.get(Proposition, pid)).text
        self.assertIn("2026-08-15", text)
        self.assertNotIn("2026-07-20", text)
        self.assertEqual(push.call_args.kwargs["observer_name"], "gumbo_agenda_edit")

    async def test_edit_upserts_single_row(self):
        pid = await self._seed_prop_with_obs("Renew the license", 1)
        with mock.patch.object(self.gum.batcher, "push"):
            await self.gum.apply_agenda_override(pid, title="Renew lab license")
            await self.gum.apply_agenda_override(pid, status="in progress")
        async with self.gum._session() as s:
            rows = (await s.execute(select(AgendaOverride))).scalars().all()
        self.assertEqual(len(rows), 1)  # merged, not duplicated
        self.assertEqual(rows[0].title, "Renew lab license")
        self.assertEqual(rows[0].status, "in progress")

    async def test_ambiguous_date_skips_text_rewrite(self):
        # Two dates in the text → the rewrite is ambiguous and must be skipped, but
        # the override + correction observation still record the correction.
        pid = await self._seed_prop_with_obs("Meet 2026-07-10, deliver 2026-07-20", 1)
        with mock.patch.object(self.gum.batcher, "push") as push:
            await self.gum.apply_agenda_override(pid, due_date="2026-08-15")
        async with self.gum._session() as s:
            text = (await s.get(Proposition, pid)).text
        self.assertNotIn("2026-08-15", text)  # text untouched
        self.assertIn("2026-07-20", text)
        self.assertEqual((await self._one_override()).due_date, "2026-08-15")  # override still set
        self.assertTrue(push.called)

    async def test_dismiss_keeps_proposition(self):
        pid = await self._seed_prop_with_obs("Reorganize the desktop icons", 1)
        with mock.patch.object(self.gum.batcher, "push") as push:
            ok = await self.gum.dismiss_agenda_item(pid)
        self.assertTrue(ok)
        self.assertTrue((await self._one_override()).dismissed)
        async with self.gum._session() as s:
            self.assertIsNotNone(await s.get(Proposition, pid))  # NOT deleted
        self.assertEqual(push.call_args.kwargs["observer_name"], "gumbo_agenda_dismiss")

    async def test_override_cascades_when_proposition_deleted(self):
        pid = await self._seed_prop_with_obs("Pay the invoice", 1)
        with mock.patch.object(self.gum.batcher, "push"):
            await self.gum.apply_agenda_override(pid, title="Pay the Q3 invoice")
        self.assertIsNotNone(await self._one_override())
        await self.gum.delete_proposition(pid)
        self.assertIsNone(await self._one_override())  # cascade removed the override

    async def test_clear_override_removes_row(self):
        pid = await self._seed_prop_with_obs("Book the venue", 1)
        with mock.patch.object(self.gum.batcher, "push"):
            await self.gum.apply_agenda_override(pid, title="Book the Chicago venue")
        self.assertTrue(await self.gum.clear_agenda_override(pid))
        self.assertIsNone(await self._one_override())
        self.assertFalse(await self.gum.clear_agenda_override(pid))  # already gone

    async def test_edit_missing_proposition_returns_false(self):
        with mock.patch.object(self.gum.batcher, "push"):
            self.assertFalse(await self.gum.apply_agenda_override(999_999, title="x"))
            self.assertFalse(await self.gum.dismiss_agenda_item(999_999))

    async def test_edit_does_not_hold_batch_lock(self):
        pid = await self._seed_prop_with_obs("stays responsive", 1)
        with mock.patch.object(self.gum.batcher, "push"):
            await self.gum.apply_agenda_override(pid, status="blocked")
        self.assertFalse(self.gum._batch_processing_lock.locked())


class AgendaItemMethodTests(_Base):
    """Explicitly-added agenda items live in their own table, separate from the
    inferred-belief propositions, and never touch the batch/inference pipeline."""

    async def _all_items(self, include_dismissed=True) -> list[AgendaItem]:
        async with self.gum._session() as s:
            stmt = select(AgendaItem)
            return list((await s.execute(stmt)).scalars().all())

    async def test_add_returns_id_and_stores_row(self):
        iid = await self.gum.add_agenda_item(
            title="Submit the Q3 report", due_date="2999-07-20",
            status="in progress", source="added by an assistant", created_by="mcp",
        )
        self.assertIsInstance(iid, int)
        items = await self._all_items()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "Submit the Q3 report")
        self.assertEqual(items[0].created_by, "mcp")
        # No proposition was created — added items are NOT inferred beliefs.
        async with self.gum._session() as s:
            self.assertEqual((await s.execute(select(Proposition))).scalars().all(), [])

    async def test_add_empty_title_returns_none(self):
        self.assertIsNone(await self.gum.add_agenda_item(title="   "))
        self.assertEqual(await self._all_items(), [])

    async def test_list_excludes_dismissed_by_default(self):
        iid = await self.gum.add_agenda_item(title="Pay invoice")
        await self.gum.add_agenda_item(title="Book venue")
        await self.gum.set_agenda_item_dismissed(iid, True)
        listed = await self.gum.list_agenda_items()
        self.assertEqual({i["title"] for i in listed}, {"Book venue"})
        listed_all = await self.gum.list_agenda_items(include_dismissed=True)
        self.assertEqual(len(listed_all), 2)

    async def test_update_edits_fields(self):
        iid = await self.gum.add_agenda_item(title="Draft", due_date="2999-07-20")
        self.assertTrue(await self.gum.update_agenda_item(
            iid, title="Draft the memo", status="blocked", due_date="2999-08-01"))
        item = (await self._all_items())[0]
        self.assertEqual(item.title, "Draft the memo")
        self.assertEqual(item.status, "blocked")
        self.assertEqual(item.due_date, "2999-08-01")

    async def test_update_clear_due_date(self):
        iid = await self.gum.add_agenda_item(title="Thing", due_date="2999-07-20")
        await self.gum.update_agenda_item(iid, clear_due_date=True)
        self.assertIsNone((await self._all_items())[0].due_date)

    async def test_update_and_dismiss_missing_returns_false(self):
        self.assertFalse(await self.gum.update_agenda_item(999_999, title="x"))
        self.assertFalse(await self.gum.set_agenda_item_dismissed(999_999, True))

    async def test_dismiss_and_restore(self):
        iid = await self.gum.add_agenda_item(title="Reversible")
        self.assertTrue(await self.gum.set_agenda_item_dismissed(iid, True))
        self.assertEqual(await self.gum.list_agenda_items(), [])
        self.assertTrue(await self.gum.set_agenda_item_dismissed(iid, False))
        self.assertEqual(len(await self.gum.list_agenda_items()), 1)

    async def test_add_does_not_hold_batch_lock(self):
        await self.gum.add_agenda_item(title="stays responsive")
        self.assertFalse(self.gum._batch_processing_lock.locked())


if __name__ == "__main__":
    unittest.main()
