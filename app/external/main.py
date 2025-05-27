# import asyncio
# import base64
# import os
# import time
# from pathlib import Path
# import json

# import cv2
# import numpy as np
# import yaml
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.responses import HTMLResponse

# from app.external.nanodet.predictor import Predictor
# from nanodet.nanodet import Logger, cfg, load_config

# import json
# import base64
# import time
# import cv2
# import numpy as np
# from pathlib import Path
# from fastapi import WebSocket, WebSocketDisconnect


# def loadConfig(path):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# app = FastAPI()
# predictor = None


# @app.on_event("startup")
# def startup():
#     global predictor

#     config = loadConfig("configs/configs.yml")
#     load_config(cfg, config["model2"]["configPath"])
#     logger = Logger(0, use_tensorboard=False)
#     predictor = Predictor(cfg, config["model2"]["modelPath"], logger, device="cpu")


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     print("Client connected")

#     try:
#         while True:
#             data = await websocket.receive_text()
#             payload = json.loads(data)
#             image_data = payload["image"]
#             location = payload.get("location", {"lat": None, "lon": None})

#             print("Lokasi:", location)

#             header, encoded = image_data.split(",", 1)
#             img_bytes = base64.b64decode(encoded)
#             np_arr = np.frombuffer(img_bytes, np.uint8)
#             img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#             # Run detection
#             meta, res = predictor.inference(img)
#             result_img = predictor.visualize(res[0], meta, cfg.class_names, 0.35)

#             # Encode result image back to base64
#             _, buffer = cv2.imencode(".jpg", result_img)
#             encoded_result = base64.b64encode(buffer).decode("utf-8")

#             # Save image locally with timestamp
#             timestamp = int(time.time() * 1000)
#             save_dir = Path("result")
#             save_dir.mkdir(exist_ok=True)
#             filename = save_dir / f"frame_{timestamp}.jpg"
#             cv2.imwrite(str(filename), result_img)

#             # Simpan lokasi ke file (opsional)
#             if location["lat"] and location["lon"]:
#                 with open(save_dir / "gps_log.txt", "a") as f:
#                     f.write(
#                         f"{filename.name}, lat={location['lat']}, lon={location['lon']}\n"
#                     )

#             await websocket.send_text(f"data:image/jpeg;base64,{encoded_result}")

#     except WebSocketDisconnect:
#         print("Client disconnected")


# @app.get("/")
# def index():
#     html = f"""
#     <!DOCTYPE html>
#     <html>
#     <head><title>NanoDet Real-time</title></head>
#     <body>
#     <h2>NanoDet Real-time Detection (Client Kamera)</h2>

#     <label for="cameraSelect">Pilih Kamera:</label>
#     <select id="cameraSelect">
#         <option value="user">Depan</option>
#         <option value="environment" selected>Belakang</option>
#     </select>
#     <br><br>

#     <video id="video" width="640" height="480" autoplay muted></video>
#     <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
#     <canvas id="output" width="640" height="480"></canvas>
#     <script>
#         const video = document.getElementById('video');
#         const canvas = document.getElementById('canvas');
#         const ctx = canvas.getContext('2d');
#         const output = document.getElementById('output');
#         const outCtx = output.getContext('2d');
#         const cameraSelect = document.getElementById('cameraSelect');

#         let currentStream;
#         let clientLocation = {{ lat: null, lon: null }};

#         // Ambil lokasi GPS user
#         if (navigator.geolocation) {{
#             navigator.geolocation.getCurrentPosition(
#                 position => {{
#                     clientLocation.lat = position.coords.latitude;
#                     clientLocation.lon = position.coords.longitude;
#                     console.log("Lokasi ditemukan:", clientLocation);
#                 }},
#                 error => {{
#                     console.warn("Gagal ambil lokasi:", error.message);
#                 }},
#                 {{ enableHighAccuracy: true }}
#             );
#         }} else {{
#             alert("Geolocation tidak didukung oleh browser.");
#         }}

#         function stopMediaTracks(stream) {{
#             stream.getTracks().forEach(track => track.stop());
#         }}

#         async function startCamera(facingMode) {{
#             if (currentStream) {{
#                 stopMediaTracks(currentStream);
#             }}

#             try {{
#                 const constraints = {{
#                     video: {{ facingMode: facingMode }}
#                 }};
#                 currentStream = await navigator.mediaDevices.getUserMedia(constraints);
#                 video.srcObject = currentStream;
#             }} catch (err) {{
#                 alert("Gagal akses kamera: " + err);
#             }}
#         }}

#         cameraSelect.addEventListener("change", () => {{
#             const selected = cameraSelect.value;
#             startCamera(selected);
#         }});

#         // Default: kamera belakang
#         startCamera("environment");

#         const ws = new WebSocket("wss://7c8b-182-253-252-146.ngrok-free.app/ws");

#         ws.onopen = () => {{
#             console.log("WebSocket terhubung");
#             sendFrame();
#         }};

#         ws.onmessage = (event) => {{
#             const img = new Image();
#             img.onload = () => {{
#                 outCtx.drawImage(img, 0, 0, output.width, output.height);
#                 setTimeout(sendFrame, 150);
#             }};
#             img.src = event.data;
#         }};

#         function sendFrame() {{
#             ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
#             const dataURL = canvas.toDataURL('image/jpeg', 0.7);

#             const payload = {{
#                 image: dataURL,
#                 location: clientLocation
#             }};

#             ws.send(JSON.stringify(payload));
#         }}
#     </script>
#     </body>
#     </html>
#     """
#     return HTMLResponse(content=html)
