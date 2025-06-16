import base64
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from fastapi import HTTPException

from ..core.config import DETECTOR_SETTINGS, UPLOAD_FILES_DIRECTORY
from ..core.database import SessionLocal
from ..external.inference.predictor import Predictor
from ..external.nanodet.nanodet.util import Logger, cfg, load_config
from ..models.report_model import *  # Untuk tipe return dan Enum jika perlu
from ..schemas.report_schema import ReportCreate
from .report_services import ReportService

detector_settings = DETECTOR_SETTINGS
model = detector_settings.model
config = detector_settings.config
webcam = detector_settings.webcam
threshold = detector_settings.threshold
device = detector_settings.device
save_detection_img = UPLOAD_FILES_DIRECTORY
save_detection_img.mkdir(parents=True, exist_ok=True)
logger = Logger(0, use_tensorboard=False)
load_config(cfg, DETECTOR_SETTINGS.config)
predictor_instance = Predictor(cfg, model, logger, device)


def decode_image_base64(encoded_str: str) -> np.ndarray:
    """Decode base64 image string (data:image/jpeg;base64,...) menjadi numpy array."""
    _, encoded = encoded_str.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def run_inference(img: np.ndarray) -> Tuple[Dict, list]:
    meta, res = predictor_instance.inference(img)
    return meta, res


def visualize_result(dets, meta, threshold: float = 0.35):
    return predictor_instance.visualize(dets, meta, cfg.class_names, threshold)


def save_image_result(img: np.ndarray, filename_prefix: str = "frame") -> Path:
    timestamp = int(time.time() * 1000)
    file_path = save_detection_img / f"{filename_prefix}_{timestamp}.jpg"
    cv2.imwrite(str(file_path), img)
    return file_path


def prepare_detection_payload(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def analyze_image(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_input is None:
        raise ValueError("Gagal membaca data gambar.")

    meta, res = run_inference(img_input)
    result_img, class_img = visualize_result(res[0], meta, threshold)

    if class_img is not None and class_img != "None":
        return prepare_detection_payload(result_img), class_img
    return None, None


async def save_report_from_detection(
    lat: Optional[float],
    lng: Optional[float],
    detected_damage_type: str,
    image_relative_url: Path,
    description_prefix: str = "Deteksi otomatis dari WS",
) -> Optional[Report]:
    """
    Fungsi helper untuk membuat dan menyimpan entri laporan kerusakan ke database.
    Fungsi ini akan membuat sesi database sendiri.
    """
    if lat is None or lng is None:
        print(
            "DB Save Helper: Data latitude atau longitude tidak lengkap, laporan tidak disimpan."
        )
        return None

    db = SessionLocal()  # Buat sesi database baru
    try:
        print(
            f"DB Save Helper: Mencoba menyimpan laporan: lat={lat}, lng={lng}, type={detected_damage_type}, photo_url={image_relative_url}"
        )

        # 1. Buat skema Pydantic ReportCreate
        # Pastikan field `damage_type` di Pydantic model Anda memiliki alias `type`
        # jika Anda ingin menggunakan `type` sebagai argumen fungsi.
        # Atau, ganti argumen fungsi menjadi `damage_type_from_ws`.
        report_create_schema = ReportCreate(
            lat=lat,
            lng=lng,
            type=detected_damage_type,  # Ini akan dipetakan ke 'damage_type' jika alias ada di ReportCreate
            description=f"{description_prefix} pada {time.strftime('%Y-%m-%d %H:%M:%S')}",
        )

        # 2. Buat instance ReportService dengan sesi DB yang baru dibuat
        report_service = ReportService(db=db)

        # 3. Panggil metode service untuk membuat laporan.
        #    Service akan memanggil repositori. Kita perlu memastikan service/repo
        #    bisa menangani kasus di mana foto sudah ada (URL-nya) dan tidak perlu di-upload ulang.

        #    Modifikasi `ReportService.create_report` dan `ReportRepository.create_report_in_db`
        #    untuk menerima argumen opsional `existing_photo_url: Optional[str] = None`.
        #    Jika `existing_photo_url` diberikan, maka `photo_file` akan diabaikan dan URL ini yang dipakai.

        # Panggil metode service yang dimodifikasi (lihat Langkah 2 di bawah)
        created_db_report_model = (
            await report_service.create_report_with_existing_photo_url(
                report_create_schema,
                "/uploads/" + str(image_relative_url.name),
            )
        )

        print(
            f"DB Save Helper: Laporan berhasil disimpan dengan ID: {created_db_report_model.id}"
        )
        return created_db_report_model

    except HTTPException as e:  # Jika service melempar HTTPException
        print(f"DB Save Helper Error (HTTPException): {e.detail}")
        # Kita tidak bisa raise HTTPException di WebSocket handler, tapi fungsi ini bisa
        # mengembalikan None atau raise error lain yang ditangani oleh pemanggil.
        # Untuk sekarang, kita biarkan error asli naik jika bukan dari proses ini sendiri.
        # Jika error dari service, biarkan naik.
        raise
    except ValueError as e:  # Error validasi Pydantic atau lainnya
        print(f"DB Save Helper Error (ValueError): {e}")
        raise
    except Exception as e:
        print(
            f"DB Save Helper Error (Umum): Terjadi error saat menyimpan laporan ke DB: {e}"
        )
        # Rollback jika ada masalah di tengah transaksi (meskipun commit ada di service/repo)
        db.rollback()
        raise  # Biarkan pemanggil (WebSocket handler) menangani error ini
    finally:
        db.close()  # Selalu tutup sesi database
        print("DB Save Helper: Sesi database ditutup.")
