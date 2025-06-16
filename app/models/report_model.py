import enum
from typing import Optional

from sqlalchemy import Column, Date
from sqlalchemy import Float, Integer, String
from sqlalchemy.sql import func

from ..core.database import Base


class Report(Base):
    __tablename__ = "damage_reports"  # Nama tabel di database

    # Kolom-kolom tabel
    id: int = Column(Integer, primary_key=True, index=True, autoincrement=True)
    lat: float = Column(Float, nullable=False)
    lng: float = Column(Float, nullable=False)
    damage_type: str = Column(String(100), index=True, nullable=False)

    # Kolom ini opsional
    description: Optional[str] = Column(String(500), nullable=True)

    # Path atau URL ke foto
    photo_url: Optional[str] = Column(String(255), nullable=True)

    # default=func.current_date() akan menggunakan tanggal saat ini dari server DB
    # nullable=False berarti kolom ini wajib diisi (DB akan mengisinya dengan default)
    date_reported: Date = Column(Date, default=func.current_date(), nullable=False)

    def __repr__(self):
        return f"<Report(id={self.id}, type='{self.damage_type}', status='{self.status.value}')>"


print(f"OK: Model SQLAlchemy 'Report' didefinisikan di {__file__}")
