import cv2
import numpy as np
from fastapi import UploadFile
import json

from ..core.config import DETECTOR_SETTINGS
from ..external.inference.predictor import Predictor
from ..external.nanodet.nanodet.util import Logger, cfg, load_config


class DamageDetectorService:
    """
    Service yang dioptimalkan untuk menangani deteksi gambar.
    """

    def __init__(self, settings: dict):
        self.settings = settings
        logger = Logger(0, use_tensorboard=False)
        load_config(cfg, settings.config)
        self.predictor = Predictor(cfg, settings.model, logger, device=settings.device)
        print("✅ Detector Service and NanoDet Model Initialized.")

    def analyze_image(self, image_bytes: bytes) -> tuple[np.ndarray, str]:
        """
        Menganalisis gambar dari bytes, mengembalikan gambar hasil visualisasi
        dan nama kelas yang terdeteksi.
        """
        # Ubah bytes menjadi array numpy yang bisa dibaca OpenCV
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_input is None:
            raise ValueError("Gagal membaca data gambar.")

        # Lakukan inferensi
        meta, res = self.predictor.inference(img_input)

        # Visualisasi hasil dan dapatkan nama kelas
        # Threshold diambil dari config
        result_img, class_text = self.predictor.visualize(
            res[0],
            meta,
            cfg.class_names,
            self.settings.threshold,
        )

        # Jika tidak ada yang terdeteksi, class_text akan None.
        # Kita bisa berikan nilai default.
        if class_text is None:
            class_text = "none"  # Atau 'unknown'

        return result_img, class_text

    def run_detection_for_analysis(self, image_bytes: bytes) -> str:
        """
        Menjalankan deteksi untuk fitur 'Analisis Manual'.
        Mengembalikan satu nama kelas dengan skor tertinggi.
        """
        detections = self._perform_inference(image_bytes)

        best_detection = "none"
        if detections:
            highest_score_detection = max(detections, key=lambda x: x["score"])
            best_detection = highest_score_detection["class_name"]
        return best_detection

    def run_realtime_detection(self, image_bytes: bytes) -> str:
        """
        Menjalankan deteksi untuk 'Real-time WebSocket Streaming'.
        Mengembalikan semua deteksi di atas ambang batas sebagai string JSON.
        Ini adalah fungsi sinkron (blocking) yang harus dijalankan di thread pool.
        """
        detections = self._perform_inference(image_bytes)
        return json.dumps({"detections": detections})

    def _perform_inference(self, image_bytes: bytes) -> list:
        """
        Metode internal untuk menjalankan inferensi dan mengekstrak hasilnya.
        """
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_input is None:
            # Mengembalikan list kosong jika gambar tidak valid
            return []

        meta, res = self.predictor.inference(img_input)

        detections = []
        # res[0] adalah kamus, dengan kunci label dan nilai daftar bbox
        if res and res[0] is not None:
            for label_idx, bboxes in res[0].items():
                for bbox in bboxes:
                    # bbox format: [x1, y1, x2, y2, score]
                    if bbox[4] > self.settings.threshold:
                        class_name = cfg.class_names[label_idx]
                        detections.append(
                            {
                                "box": [int(coord) for coord in bbox[:4]],
                                "score": float(bbox[4]),
                                "class_name": class_name,
                            }
                        )
        return detections


detector_service = DamageDetectorService(settings=DETECTOR_SETTINGS)


# Buat dependency yang bisa digunakan di router
def get_detector_service() -> DamageDetectorService:
    return detector_service
