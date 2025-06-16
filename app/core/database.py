from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session as SessionType
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import DATABASE_URL

# Jika menggunakan SQLite, tambahkan connect_args
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # pool_pre_ping baik untuk DB eksternal
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base untuk model SQLAlchemy
Base = declarative_base()


# Dependency untuk mendapatkan sesi database di endpoint/service
def get_db_session() -> Generator[SessionType, None, None]:
    db: SessionType = SessionLocal()
    try:
        yield db
    finally:
        db.close()


print(
    f"OK: Database engine dan SessionLocal dikonfigurasi untuk '{DATABASE_URL[:20]}...'."
)
