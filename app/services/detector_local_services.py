import cv2
import asyncio
import threading

from ..core.locaton_store import get_last_location
from .detector_services import (
    webcam,
    threshold,
    run_inference,
    save_image_result,
    save_report_from_detection,
    threshold,
    visualize_result,
)


class LocalDetection:
    def __init__(self):
        self.running = False
        self.detector_thread = None
        self.webcam = webcam

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
            print("Kamera tidak bisa dibuka.")
            return

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                location = get_last_location()
                meta, res = run_inference(frame)
                result_img, class_img = visualize_result(res[0], meta, threshold)

                if class_img is not None and class_img != "None":
                    file_path = save_image_result(result_img, filename_prefix="local")
                    lat = location.get("lat")
                    lng = location.get("lon")
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            save_report_from_detection(lat, lng, class_img, file_path)
                        )
                    except Exception as e:
                        print(f"Gagal simpan laporan: {e}")
                    finally:
                        loop.close()

                    print(f"📍 Lokasi: {get_last_location()}")
                    print(f"🧠 Deteksi: {class_img}")
                cv2.waitKey(1)

        except Exception as e:
            print("Error:", e)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Kamera dilepas.")
