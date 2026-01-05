"""
Database Connection Management
"""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from config import settings

# Create async engine
# Use sqlite for development if no DATABASE_URL provided, but warn about it
database_url = settings.database_url
if not database_url:
    # Fallback for local dev if not set (though config.py validates this in prod)
    database_url = "sqlite+aiosqlite:///./finpilot.db"
    print(f"⚠️  WARNING: No DATABASE_URL set. Using local SQLite: {database_url}")

engine = create_async_engine(
    database_url,
    echo=settings.database_echo,
    pool_pre_ping=True,
    # SQLite specific args
    connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)


class Base(DeclarativeBase):
    """Base class for all ORM models"""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database sessions.
    Yields an AsyncSession and ensures it's closed after use.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database (create tables) - for dev/testing only"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
