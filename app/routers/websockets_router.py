# app/routers/websockets_router.py
from fastapi import (
    APIRouter,
    Depends,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from ..services.detector_services import DamageDetectorService, get_detector_service

router = APIRouter(tags=["WebSocket Detection"])


@router.websocket("")
async def websocket_detection_endpoint(
    websocket: WebSocket,
    detector: DamageDetectorService = Depends(get_detector_service),
):
    """
    Endpoint WebSocket yang dioptimalkan untuk deteksi real-time.
    - Menerima: Frame gambar dalam format biner (image/jpeg).
    - Menjalankan: Deteksi di thread terpisah agar tidak memblokir server.
    - Mengirim: Hasil deteksi (koordinat, kelas, skor) dalam format JSON.
    """
    await websocket.accept()
    print("WS Client terhubung ke endpoint deteksi real-time.")

    try:
        while True:
            # 1. Terima data frame gambar sebagai byte
            image_bytes = await websocket.receive_bytes()

            # 2. Jalankan fungsi deteksi yang berat di thread pool
            # Ini adalah kunci untuk menjaga server tetap responsif.
            results_json_str = await run_in_threadpool(
                detector.run_realtime_detection, image_bytes
            )

            # 3. Kirim kembali hasilnya sebagai teks JSON yang ringan
            await websocket.send_text(results_json_str)

    except WebSocketDisconnect:
        print("WS Client terputus.")
    except Exception as e:
        print(f"WS Error tidak terduga: {e}")
        # Pastikan koneksi ditutup jika terjadi error
        if websocket.client_state != WebSocketDisconnect:
            await websocket.close(code=1011)
