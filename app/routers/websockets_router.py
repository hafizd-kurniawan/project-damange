import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..services.detector_services import (
    decode_image_base64,
    predictor_instance,
    prepare_detection_payload,
    run_inference,
    save_image_result,
    save_report_from_detection,
    threshold,
    visualize_result,
)
from ..services.report_services import ReportService

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

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)

                image_data = payload.get("image")
                location = payload.get("location", {"lat": None, "lon": None})

                if not image_data:
                    await websocket.send_text("Invalid image data")
                    continue

                img = decode_image_base64(image_data)
                if img is None:
                    await websocket.send_text("Failed to decode image")
                    continue

                meta, res = run_inference(img)
                result_img, class_img = visualize_result(res[0], meta, threshold)
                if class_img is not None and class_img != "None":
                    file_path = save_image_result(result_img, filename_prefix="ws")
                    lat = location.get("lat")
                    lng = location.get("lon")
                    await save_report_from_detection(lat, lng, class_img, file_path)

                encoded_result = prepare_detection_payload(result_img)
                await websocket.send_text(f"data:image/jpeg;base64,{encoded_result}")

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
