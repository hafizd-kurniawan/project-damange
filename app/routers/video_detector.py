# app/routers/video_detector.py
import cv2
import time
from pathlib import Path
import threading
from ..core.locaton_store import get_last_location
from ..core.database import SessionLocal
from ..schemas.report_schema import ReportCreate, DamageSeverityEnum
from ..services.report_services import ReportService
from ..external.inference.predictor import Predictor
from ..core.config import UPLOAD_FILES_DIRECTORY, DETECTOR_SETTINGS
from ..core.config import DETECTOR_SETTINGS, UPLOAD_FILES_DIRECTORY
from ..core.database import SessionLocal  # Untuk membuat sesi DB baru

# --- Modifikasi Impor dan Inisialisasi Model NanoDet ---
# Asumsikan struktur dan cara kerja modul eksternal Anda
# try:
# Impor kelas Predictor dan fungsi/variabel konfigurasi yang relevan
from ..external.inference.predictor import Predictor  # Ini harusnya kelas
from ..external.nanodet.nanodet.util import (  # Sesuaikan path jika perlu
    Logger,
    cfg,
    load_config,
)
from ..models.report_model import *  # Untuk tipe return dan Enum jika perlu
from ..schemas.report_schema import ReportCreate  # Skema Pydantic untuk membuat laporan
from ..services.report_services import ReportService

from multiprocessing import Process, Event
from .websockets_router import save_report_from_detection
import asyncio  # Tambahkan di bagian import


class LocalDetection:
    def __init__(self):
        self.process: Process = None
        self.stop_event = Event()
        self.detector_settings = DETECTOR_SETTINGS
        self.model = self.detector_settings.model
        self.config = self.detector_settings.config
        self.webcam = self.detector_settings.webcam
        self.threshold = self.detector_settings.threshold
        self.device = self.detector_settings.device
        self.logger = Logger(0, use_tensorboard=False)
        self.cfg = cfg
        self.predictor_instance = Predictor(
            self.cfg, self.model, self.logger, self.device
        )
        self.ws_result_save_dir = UPLOAD_FILES_DIRECTORY
        self.ws_result_save_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        if self.process and self.process.is_alive():
            return {"status": "already_running"}

        self.stop_event.clear()
        self.process = Process(
            target=self._run_local_detection, args=(self.stop_event,)
        )
        self.process.start()
        return {"status": "started"}

    def stop(self):
        if self.process and self.process.is_alive():
            self.stop_event.set()
            self.process.join(timeout=5)
            return {"status": "stopped"}
        return {"status": "not_running"}

    def _run_local_detection(self, stop_event):
        cap = cv2.VideoCapture(DETECTOR_SETTINGS.webcam)
        if not cap.isOpened():
            print("❌ Kamera tidak bisa dibuka.")
            return

        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                location = get_last_location()
                meta, res = self.predictor_instance.inference(frame)
                result_img, class_text = self.predictor_instance.visualize(
                    res[0], meta, self.cfg.class_names, DETECTOR_SETTINGS.threshold
                )

                if class_text is not None and class_text != "None":
                    timestamp = int(time.time() * 1000)
                    result_filename = (
                        self.ws_result_save_dir / f"detected_frame_{timestamp}.jpg"
                    )

                    cv2.imwrite(str(result_filename), result_img)
                    with open(self.ws_result_save_dir / "ws_gps_log.txt", "a") as f:
                        f.write(
                            f"{result_filename.name}, lat={location['lat']}, lon={location['lon']}\n"
                        )

                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            save_report_from_detection(
                                lat=location.get("lat"),
                                lng=location.get("lon"),
                                detected_damage_type=class_text,
                                image_relative_url="/uploads/" + result_filename.name,
                            )
                        )
                    except Exception as e:
                        print(f"❌ Gagal simpan laporan: {e}")
                    finally:
                        loop.close()

                    print(f"📍 Lokasi: {get_last_location()}")
                    print(f"🧠 Deteksi: {class_text}")

                cv2.waitKey(1)

        except Exception as e:
            print("🔥 Error:", e)
        finally:
            cap.release()
            print("✅ Kamera dilepas.")
