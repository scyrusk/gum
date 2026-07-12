# models.py

from __future__ import annotations

import pathlib
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    event,
    text as sql_text,
)
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.sql import func

class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all database models.
    
    This class provides the foundation for all SQLAlchemy models in the application,
    including async support and declarative base functionality.
    """
    pass

observation_proposition = Table(
    "observation_proposition",
    Base.metadata,
    Column(
        "observation_id",
        Integer,
        ForeignKey("observations.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "proposition_id",
        Integer,
        ForeignKey("propositions.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)




class Observation(Base):
    """Represents an observation of user behavior.
    
    This model stores observations made by various observers about user behavior,
    including the content of the observation and metadata about when and how it was made.

    Attributes:
        id (int): Primary key for the observation.
        observer_name (str): Name of the observer that made this observation.
        content (str): The actual content of the observation.
        content_type (str): Type of content (e.g., 'text', 'image', etc.).
        created_at (datetime): When the observation was created.
        updated_at (datetime): When the observation was last updated.
        propositions (set[Proposition]): Set of propositions related to this observation.
    """
    __tablename__ = "observations"

    id:            Mapped[int]   = mapped_column(primary_key=True)
    observer_name: Mapped[str]   = mapped_column(String(100), nullable=False)
    content:       Mapped[str]   = mapped_column(Text,        nullable=False)
    content_type:  Mapped[str]   = mapped_column(String(50),  nullable=False)

    created_at:    Mapped[str]   = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at:    Mapped[str]   = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    propositions: Mapped[set["Proposition"]] = relationship(
        "Proposition",
        secondary=observation_proposition,
        back_populates="observations",
        collection_class=set,
        passive_deletes=True,
        lazy="selectin",
    )

    def __repr__(self) -> str:
        """String representation of the observation.
        
        Returns:
            str: A string representation showing the observation ID and observer name.
        """
        return f"<Observation(id={self.id}, observer={self.observer_name})>"


class Proposition(Base):
    """Represents a proposition about user behavior.
    
    This model stores propositions generated from observations, including the proposition
    text, reasoning behind it, and metadata about its creation and relationships.

    Attributes:
        id (int): Primary key for the proposition.
        text (str): The actual proposition text.
        reasoning (str): The reasoning behind this proposition.
        confidence (Optional[int]): Confidence level in this proposition.
        decay (Optional[int]): Decay factor for this proposition.
        created_at (datetime): When the proposition was created.
        updated_at (datetime): When the proposition was last updated.
        revision_group (str): Group identifier for related proposition revisions.
        version (int): Version number of this proposition.

        observations (set[Observation]): Set of observations related to this proposition.
    """
    __tablename__ = "propositions"

    id:         Mapped[int]           = mapped_column(primary_key=True)
    text:       Mapped[str]           = mapped_column(Text, nullable=False)
    reasoning:  Mapped[str]           = mapped_column(Text, nullable=False)
    confidence: Mapped[Optional[int]]
    decay:      Mapped[Optional[int]]

    created_at: Mapped[str]           = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[str]           = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    revision_group: Mapped[str]       = mapped_column(String(36), nullable=False, index=True)
    version:        Mapped[int]       = mapped_column(Integer, server_default="1", nullable=False)



    observations: Mapped[set[Observation]] = relationship(
        "Observation",
        secondary=observation_proposition,
        back_populates="propositions",
        collection_class=set,
        passive_deletes=True,
        lazy="selectin",
    )

    def __repr__(self) -> str:
        """String representation of the proposition.
        
        Returns:
            str: A string representation showing the proposition ID and a preview of its text.
        """
        preview = (self.text[:27] + "…") if len(self.text) > 30 else self.text
        return f"<Proposition(id={self.id}, text={preview})>"


# Allowed ratings for a proposition review.
FEEDBACK_RATINGS = ("accurate", "partial", "inaccurate")


class PropositionFeedback(Base):
    """A user's judgment about a proposition, with optional context.

    Stores a *snapshot* of the proposition text and reasoning (not just a
    foreign key) so the judgment survives later revision or deletion of the
    proposition and can be replayed as a few-shot calibration example for the
    proposition generator. ``proposition_id`` is kept (nullable) so the review
    queue can skip propositions that have already been judged.

    Attributes:
        id (int): Primary key.
        proposition_id (Optional[int]): Source proposition, or NULL if it was
            since deleted/revised.
        proposition_text (str): Snapshot of the proposition text.
        reasoning (Optional[str]): Snapshot of the proposition reasoning.
        rating (str): One of ``FEEDBACK_RATINGS`` — "accurate", "partial"
            (somewhat accurate), or "inaccurate".
        note (Optional[str]): Free-text context the user optionally provided,
            fed back to the model as calibration guidance.
        created_at (datetime): When the judgment was made.
    """
    __tablename__ = "proposition_feedback"

    id:               Mapped[int]           = mapped_column(primary_key=True)
    proposition_id:   Mapped[Optional[int]] = mapped_column(
        ForeignKey("propositions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    proposition_text: Mapped[str]           = mapped_column(Text, nullable=False)
    reasoning:        Mapped[Optional[str]] = mapped_column(Text)
    rating:           Mapped[str]           = mapped_column(String(20), nullable=False)
    note:             Mapped[Optional[str]] = mapped_column(Text)

    created_at:       Mapped[str]           = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<PropositionFeedback(id={self.id}, rating={self.rating})>"


class AgendaOverride(Base):
    """A user's manual correction to a generated agenda item (GUMBO Agenda page).

    The agenda is *regenerated* by the local model on every request
    (:func:`gum.agenda.build_agenda`), so it has no row of its own to edit. This
    table persists the user's direct edits — a corrected title / due date /
    status, or a dismissal — keyed by the source proposition, and the REST
    ``/agenda`` route overlays them on top of each freshly-extracted radar so an
    edit sticks visually even before the correction has propagated back into the
    model through re-inference.

    A ``None`` field means "not overridden" (fall through to the model's value);
    ``due_date_cleared`` distinguishes an explicit "no fixed date" from "date not
    overridden". ``dedupe_key`` is a normalized-title snapshot
    (:func:`gum.agenda._dedupe_key`) used as a *fallback* match: the GUM's
    SIMILAR→revise path deletes-and-replaces a proposition with a **new** id,
    which cascades this row away by ``proposition_id``; matching on the stable
    normalized title lets a live override re-bind to the replacement proposition
    the next time the radar surfaces it.

    Auto-created by :func:`init_db` via ``Base.metadata.create_all`` (same as
    :class:`PropositionFeedback`); no FTS index is needed.

    Attributes:
        id (int): Primary key.
        proposition_id (int): Source proposition — unique, one override per
            proposition (repeated edits merge onto the same row). Cascade-deleted
            with the proposition.
        dedupe_key (Optional[str]): Normalized-title snapshot for fallback match.
        title (Optional[str]): Overridden title, or None.
        status (Optional[str]): Overridden status guess, or None.
        due_date (Optional[str]): Overridden ISO ``YYYY-MM-DD`` due date, or None.
        due_date_cleared (bool): True if the user explicitly set "no fixed date".
        dismissed (bool): True if the user removed the item from the radar.
        created_at / updated_at (datetime): Bookkeeping timestamps.
    """
    __tablename__ = "agenda_overrides"

    id:               Mapped[int]           = mapped_column(primary_key=True)
    proposition_id:   Mapped[int]           = mapped_column(
        ForeignKey("propositions.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    dedupe_key:       Mapped[Optional[str]] = mapped_column(String(200), index=True)
    title:            Mapped[Optional[str]] = mapped_column(Text)
    status:           Mapped[Optional[str]] = mapped_column(String(40))
    due_date:         Mapped[Optional[str]] = mapped_column(String(10))
    due_date_cleared: Mapped[bool]          = mapped_column(
        Boolean, server_default="0", nullable=False
    )
    dismissed:        Mapped[bool]          = mapped_column(
        Boolean, server_default="0", nullable=False
    )

    created_at:       Mapped[str]           = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at:       Mapped[str]           = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<AgendaOverride(id={self.id}, proposition_id={self.proposition_id}, "
            f"dismissed={self.dismissed})>"
        )


class AgendaItem(Base):
    """An agenda entry the user or an assistant added explicitly (not inferred).

    The rest of the agenda is *extracted* from the GUM's inferred propositions;
    this table holds items someone put on the agenda directly — most importantly
    ones a frontier agent adds through the MCP ``add_agenda_item`` tool. Those are
    kept in their own table, deliberately NOT as :class:`Proposition` rows, so an
    assistant's todo never masquerades as an inferred belief about the user (which
    would pollute confidence, feedback calibration, and future inference).

    Because an MCP agent only ever sees pseudonymized context, the ``title`` /
    ``note`` it sends back may carry pseudo-IDs (``[PERSON_1]``); they are
    **rehydrated to the real values on-device** (see ``Sanitizer.rehydrate``)
    before being stored here, so the local agenda shows real names. The rehydrated
    text is never returned to the agent.

    The REST ``/agenda`` route merges these in as first-class commitments (with an
    ``item_id`` instead of a ``proposition_id``); ``dismissed`` mirrors the
    override soft-hide so the UI's dismiss/undo works uniformly. Auto-created by
    :func:`init_db`.

    Attributes:
        id (int): Primary key (the agenda item's ``item_id``).
        title (str): The commitment text, stored rehydrated (real values).
        due_date (Optional[str]): ISO ``YYYY-MM-DD`` deadline, or None if undated.
        status (Optional[str]): Status guess ('not started'/'in progress'/…).
        source (Optional[str]): Provenance label shown in the UI (e.g. the agent).
        note (Optional[str]): Optional extra context, stored rehydrated.
        created_by (str): 'mcp' (an assistant) or 'user'.
        dismissed (bool): True if the user removed it from the radar.
        created_at / updated_at (datetime): Bookkeeping timestamps.
    """
    __tablename__ = "agenda_items"

    id:         Mapped[int]           = mapped_column(primary_key=True)
    title:      Mapped[str]           = mapped_column(Text, nullable=False)
    due_date:   Mapped[Optional[str]] = mapped_column(String(10))
    status:     Mapped[Optional[str]] = mapped_column(String(40))
    source:     Mapped[Optional[str]] = mapped_column(Text)
    note:       Mapped[Optional[str]] = mapped_column(Text)
    created_by: Mapped[str]           = mapped_column(String(20), server_default="user", nullable=False)
    dismissed:  Mapped[bool]          = mapped_column(Boolean, server_default="0", nullable=False)

    created_at: Mapped[str]           = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[str]           = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        preview = (self.title[:27] + "…") if len(self.title) > 30 else self.title
        return f"<AgendaItem(id={self.id}, title={preview}, by={self.created_by})>"


FTS_TOKENIZER = "porter ascii"

def create_fts_table(conn) -> None:
    """Create FTS5 virtual table and triggers for proposition search.
    
    This function creates a full-text search table for propositions and sets up
    triggers to maintain the search index as propositions are modified.

    Args:
        conn: SQLite database connection.
    """
    exists = conn.execute(
        sql_text(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='propositions_fts'"
        )
    ).fetchone()
    if exists:
        return

    conn.execute(
        sql_text(
            f"""
            CREATE VIRTUAL TABLE propositions_fts
            USING fts5(
                text,
                reasoning,
                content='propositions',
                content_rowid='id',
                tokenize='{FTS_TOKENIZER}'
            );
        """
        )
    )
    conn.execute(
        sql_text(
            """
            CREATE TRIGGER propositions_ai
            AFTER INSERT ON propositions BEGIN
                INSERT INTO propositions_fts(rowid, text, reasoning)
                VALUES (new.id, new.text, new.reasoning);
            END;
        """
        )
    )
    conn.execute(
        sql_text(
            """
            CREATE TRIGGER propositions_ad
            AFTER DELETE ON propositions BEGIN
                INSERT INTO propositions_fts(propositions_fts, rowid, text, reasoning)
                VALUES('delete', old.id, old.text, old.reasoning);
            END;
        """
        )
    )
    conn.execute(
        sql_text(
            """
            CREATE TRIGGER propositions_au
            AFTER UPDATE ON propositions BEGIN
                INSERT INTO propositions_fts(propositions_fts, rowid, text, reasoning)
                VALUES('delete', old.id, old.text, old.reasoning);
                INSERT INTO propositions_fts(rowid, text, reasoning)
                VALUES(new.id, new.text, new.reasoning);
            END;
        """
        )
    )
    conn.execute(
        sql_text(
            """
            INSERT INTO propositions_fts(rowid, text, reasoning)
            SELECT id, text, reasoning FROM propositions;
        """
        )
    )

def create_observations_fts(conn) -> None:
    """Create FTS5 virtual table and triggers for observation search.
    
    This function creates a full-text search table for observations and sets up
    triggers to maintain the search index as observations are modified.

    Args:
        conn: SQLite database connection.
    """
    exists = conn.execute(sql_text(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name='observations_fts'"
    )).fetchone()
    if exists:
        return                      # already present

    conn.execute(sql_text(f"""
        CREATE VIRTUAL TABLE observations_fts
        USING fts5(
            content,
            content='observations',
            content_rowid='id',
            tokenize='{FTS_TOKENIZER}'
        );
    """))
    conn.execute(sql_text("""
        CREATE TRIGGER observations_ai
        AFTER INSERT ON observations BEGIN
            INSERT INTO observations_fts(rowid, content)
            VALUES (new.id, new.content);
        END;
    """))
    conn.execute(sql_text("""
        CREATE TRIGGER observations_ad
        AFTER DELETE ON observations BEGIN
            INSERT INTO observations_fts(observations_fts, rowid, content)
            VALUES ('delete', old.id, old.content);
        END;
    """))
    conn.execute(sql_text("""
        CREATE TRIGGER observations_au
        AFTER UPDATE ON observations BEGIN
            INSERT INTO observations_fts(observations_fts, rowid, content)
            VALUES ('delete', old.id, old.content);
            INSERT INTO observations_fts(rowid, content)
            VALUES (new.id, new.content);
        END;
    """))
    # back-fill the index
    conn.execute(sql_text("""
        INSERT INTO observations_fts(rowid, content)
        SELECT id, content FROM observations;
    """))


def migrate_feedback_table(conn) -> None:
    """Evolve ``proposition_feedback`` from the old boolean ``verdict`` schema to
    the ``rating`` + ``note`` schema, preserving existing judgments. Idempotent.

    SQLite can't relax the old ``verdict NOT NULL`` column in place, so when the
    old schema is detected we rebuild the table (copying rows, mapping
    verdict→rating) rather than dropping any data.
    """
    cols = {
        row[1]
        for row in conn.execute(
            sql_text("PRAGMA table_info(proposition_feedback)")
        ).fetchall()
    }
    if not cols:
        return  # table doesn't exist yet; create_all will build the new schema
    if "rating" in cols:
        if "note" not in cols:
            conn.execute(sql_text("ALTER TABLE proposition_feedback ADD COLUMN note TEXT"))
        return

    # Old schema (verdict BOOLEAN NOT NULL): rebuild, mapping verdict -> rating.
    conn.execute(sql_text(
        """
        CREATE TABLE proposition_feedback__new (
            id INTEGER NOT NULL PRIMARY KEY,
            proposition_id INTEGER,
            proposition_text TEXT NOT NULL,
            reasoning TEXT,
            rating VARCHAR(20) NOT NULL,
            note TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
            FOREIGN KEY(proposition_id) REFERENCES propositions (id) ON DELETE SET NULL
        )
        """
    ))
    conn.execute(sql_text(
        """
        INSERT INTO proposition_feedback__new
            (id, proposition_id, proposition_text, reasoning, rating, note, created_at)
        SELECT id,
               CASE WHEN proposition_id IN (SELECT id FROM propositions)
                    THEN proposition_id ELSE NULL END,
               proposition_text, reasoning,
               CASE WHEN verdict THEN 'accurate' ELSE 'inaccurate' END, NULL, created_at
        FROM proposition_feedback
        """
    ))
    conn.execute(sql_text("DROP TABLE proposition_feedback"))
    conn.execute(sql_text("ALTER TABLE proposition_feedback__new RENAME TO proposition_feedback"))
    conn.execute(sql_text(
        "CREATE INDEX IF NOT EXISTS ix_proposition_feedback_proposition_id "
        "ON proposition_feedback (proposition_id)"
    ))


async def init_db(
    db_path: str = "gum.db",
    db_directory: Optional[str] = None,
):
    """Create the SQLite file, ORM tables & FTS5 index (first run only)."""
    if db_directory:
        path = pathlib.Path(db_directory).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        db_path = str(path / db_path)

    engine: AsyncEngine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        future=True,
        connect_args={
            "timeout": 30,
            "isolation_level": None,
        },
        poolclass=None,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.execute(sql_text("PRAGMA journal_mode=WAL"))
        await conn.execute(sql_text("PRAGMA busy_timeout=30000"))

        await conn.run_sync(migrate_feedback_table)
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(create_fts_table)
        await conn.run_sync(create_observations_fts)

    Session = async_sessionmaker(
        engine, 
        expire_on_commit=False,
        autoflush=False
    )
    return engine, Session
