"""Alembic environment.

Reads the database URL from the application settings rather than alembic.ini, so
there is one place a connection string is configured and no chance of migrations
running against a different database than the app.

Autogenerate is configured to compare types and server defaults, because the
default settings miss exactly the changes that cause a production incident: a
column widened in the model but not in the database, or a default that exists in
one and not the other.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from src.core.config import get_settings

# Importing the package registers every mapper; a model module missing from
# src/models/__init__.py is silently absent from migrations.
from src.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)


def include_object(object_, name, type_, reflected, compare_to) -> bool:
    """Decide whether autogenerate should manage an object.

    The pgvector HNSW indexes are created by an explicit operation in the initial
    migration, because Alembic cannot express their build parameters. Excluding
    them here stops every subsequent autogenerate from proposing to drop and
    recreate them.
    """
    return not (type_ == "index" and name and name.endswith("_hnsw"))


def run_migrations_offline() -> None:
    """Emit SQL without connecting, for review in a change process."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations on an open connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
        # Migrations run in one transaction so a failure half way leaves the
        # schema as it was rather than in an undefined intermediate state.
        transaction_per_migration=False,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Connect and run migrations."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migrations."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
