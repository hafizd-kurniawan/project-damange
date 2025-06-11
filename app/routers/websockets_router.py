# app/routers/websockets_router.py

import base64
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)

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

detector_settings = DETECTOR_SETTINGS
model = detector_settings.model
config = detector_settings.config
webcam = detector_settings.webcam
threshold = detector_settings.threshold
device = detector_settings.device
logger = Logger(0, use_tensorboard=False)

load_config(cfg, config)
predictor_instance = Predictor(cfg, model, logger, device)


router = APIRouter(tags=["WebSocket Detection"])


@router.websocket("")
async def websocket_detection_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WS Client terhubung ke /ws/detect")

    if not predictor_instance:  # Cek apakah predictor berhasil diinisialisasi
        print("WS Error: Predictor tidak terinisialisasi. Menutup koneksi.")
        await websocket.send_text(
            json.dumps({"error": "Detection model is not available."})
        )
        await websocket.close(code=1011)
        return

    ws_result_save_dir = UPLOAD_FILES_DIRECTORY
    ws_result_save_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            data = await websocket.receive_text()
            # ... (sisa logika parsing JSON, decode base64 seperti sebelumnya) ...
            try:
                # data = image_to_base64_data_url()
                payload = json.loads(data)

                image_data_base64 = payload.get("image")
                location = payload.get("location", {"lat": None, "lon": None})

                if not image_data_base64:  # dst. (seperti kode Anda)
                    await websocket.send_text(
                        json.dumps({"error": "No image data in payload"})
                    )
                    continue

                header, encoded_image_str = image_data_base64.split(",", 1)
                img_bytes = base64.b64decode(encoded_image_str)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img_input_for_detection = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img_input_for_detection is None:
                    await websocket.send_text(
                        json.dumps({"error": "Failed to decode image."})
                    )
                    continue
            except Exception as e:
                await websocket.send_text(
                    json.dumps({"error": f"Error processing input: {str(e)}"})
                )
                continue

            # --- Jalankan Deteksi menggunakan instance predictor ---
            try:
                # Sekarang panggil metode pada instance predictor_instance
                meta, res = predictor_instance.inference(img_input_for_detection)
                result_img_visualized, class_text = predictor_instance.visualize(
                    res[0],
                    meta,
                    cfg.class_names,
                    threshold,
                )
            except Exception as e:
                print(f"WS Error saat inferensi atau visualisasi: {e}")
                await websocket.send_text(
                    json.dumps(
                        {"error": f"Error during detection/visualization: {str(e)}"}
                    )
                )
                continue
            # --- Akhir Deteksi ---

            # ... (sisa logika encode hasil, simpan file, kirim ke klien seperti sebelumnya) ...
            try:
                _, buffer = cv2.imencode(".jpg", result_img_visualized)
                encoded_result_str = base64.b64encode(buffer).decode("utf-8")
                timestamp = int(time.time() * 1000)

                if class_text is not None and class_text != "None":
                    result_filename = (
                        ws_result_save_dir / f"detected_frame_{timestamp}.jpg"
                    )
                    cv2.imwrite(str(result_filename), result_img_visualized)
                    with open(ws_result_save_dir / "ws_gps_log.txt", "a") as f:
                        f.write(
                            f"{result_filename.name}, lat={location['lat']}, lon={location['lon']}\n"
                        )

                    await save_report_from_detection(
                        location.get("lat"),
                        location.get("lon"),
                        class_text,
                        "/uploads/" + result_filename.name,
                    )

                await websocket.send_text(
                    f"data:image/jpeg;base64,{encoded_result_str}"
                )

            except Exception as e:
                print(f"WS Error saat memproses output atau mengirim: {e}")
                await websocket.send_text(
                    json.dumps({"error": f"Error processing/sending result: {str(e)}"})
                )
                continue
    except WebSocketDisconnect as e:
        print("WS Client terputus dari /ws/detect", e)
    except Exception as e:
        print(f"WS Error tidak terduga di endpoint /ws/detect: {e}")
        if websocket.client_state != WebSocketDisconnect:
            try:
                await websocket.close(code=1011)
            except RuntimeError:
                pass
    finally:
        print("WS: Menutup koneksi /ws/detect (jika masih ada).")


@router.post("/detect-image")
async def detect_image(request):
    payload = await request.json()
    image_data_base64 = payload.get("image")
    location = payload.get("location", {"lat": None, "lon": None})

    if not image_data_base64:
        raise HTTPException(status_code=400, detail="Image missing")

    # Decode base64
    header, encoded = image_data_base64.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Detection
    meta, res = predictor_instance.inference(img)
    result_img = predictor_instance.visualize(res[0], meta, cfg.class_names, 0.35)

    # Save image
    timestamp = int(time.time() * 1000)
    file_path = UPLOAD_FILES_DIRECTORY / f"frame_http_{timestamp}.jpg"
    cv2.imwrite(str(file_path), result_img)

    # Save to DB (seperti biasa)
    db = SessionLocal()
    report_service = ReportService(db=db)
    report_create = ReportCreate(
        lat=location.get("lat", 0.0),
        lng=location.get("lon", 0.0),
        type="road",  # atau infer dari klasifikasi
        severity=DamageSeverityEnum.medium,
    )
    report_service.create_report_from_camera(report_create, str(file_path))
    db.close()

    # Encode result to base64 for frontend
    _, buffer = cv2.imencode(".jpg", result_img)
    encoded_result = base64.b64encode(buffer).decode("utf-8")

    return {"result_image": f"data:image/jpeg;base64,{encoded_result}"}


PATH_TO_YOUR_IMAGE = Path(
    "media_uploads/ws_detections/detected_frame_1748009805500.jpg"
)
# Anda mungkin perlu membuat direktori app/static dan meletakkan gambar di sana.

# 2. Koordinat GPS Dummy
DUMMY_LOCATION = {"lat": -6.2000, "lon": 106.8000}  # Ganti dengan koordinat dummy Anda


# 3. Fungsi untuk membaca gambar lokal dan mengubahnya ke base64 (seperti yang dikirim klien)
def image_to_base64_data_url():
    with open(PATH_TO_YOUR_IMAGE, "rb") as f:
        img_bytes = f.read()

    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{base64_str}"

    # Lokasi dummy (bisa kamu ubah)
    location = {"lat": -6.200000, "lon": 106.816666}

    payload = {"image": data_url, "location": location}

    return json.dumps(payload)


async def save_report_from_detection(
    lat: Optional[float],
    lng: Optional[
        float
    ],  # Pastikan klien mengirim 'lng' jika ini yang dipakai, atau 'lon'
    detected_damage_type: str,
    image_relative_url: str,  # URL relatif gambar yang sudah disimpan, misal "/uploads/ws_detections/file.jpg"
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
            severity=DamageSeverityEnum.medium,  # Default, atau bisa dari hasil deteksi jika ada
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
                str(image_relative_url),  # Teruskan URL foto yang sudah disimpan
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
