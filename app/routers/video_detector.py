import cv2
import time
import asyncio
import threading
from pathlib import Path

from ..core.locaton_store import get_last_location
from ..external.inference.predictor import Predictor
from ..external.nanodet.nanodet.util import Logger, cfg
from ..core.config import DETECTOR_SETTINGS, UPLOAD_FILES_DIRECTORY
from .websockets_router import save_report_from_detection


class LocalDetection:
    def __init__(self):
        self.running = False
        self.detector_thread = None
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
        if self.detector_thread and self.detector_thread.is_alive():
            return {"status": "already_running"}

        self.running = True
        self.detector_thread = threading.Thread(target=self._run_local_detection)
        self.detector_thread.start()
        return {"status": "started"}

    def stop(self):
        if self.running:
            self.running = False
            self.detector_thread.join(timeout=5)
            return {"status": "stopped"}
        return {"status": "not_running"}

    def _run_local_detection(self):
        cap = cv2.VideoCapture(self.webcam)
        if not cap.isOpened():
            print("❌ Kamera tidak bisa dibuka.")
            return

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                location = get_last_location()
                meta, res = self.predictor_instance.inference(frame)
                result_img, class_text = self.predictor_instance.visualize(
                    res[0], meta, self.cfg.class_names, self.threshold
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
