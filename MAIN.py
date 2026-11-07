from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
import httpx
import os
import uvicorn
from ultralytics import YOLO

app = FastAPI(
    title="RoadGuard & BharatCode AI Backend",
    version="2.4.0",
    description="Standalone Python FastAPI backend integrating YOLOv8 pothole detection and BharatCode API."
)

# --------------------------------------------------------------------------
# 1. CORS CONFIGURATION (Allows your frontend to communicate with this backend)
# --------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# 2. BHARATCODE AI API CONFIGURATION (From your documentation screenshots)
# --------------------------------------------------------------------------
BHARATCODE_BASE_URL = os.getenv("BHARATCODE_BASE_URL", "https://bharatcode.ai/api/model/v1")
BHARATCODE_API_KEY = os.getenv("BHARATCODE_API_KEY", "bc_live_your_key_here")

# Supported models from your BharatCode catalog:
# - "qwen-3.8-27b"
# - "deepseek-v4.1-flash"
DEFAULT_MODEL = "qwen-3.8-27b"

class BharatCodeRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL
    input: str
    store: Optional[bool] = False
    stream: Optional[bool] = False

# --------------------------------------------------------------------------
# 3. LOCAL COMPUTER VISION MODEL INITIALIZATION (YOLOv8)
# --------------------------------------------------------------------------
try:
    cv_model = YOLO("pothole_yolov8n.pt")
except Exception:
    try:
        cv_model = YOLO("yolov8n.pt")
    except Exception:
        cv_model = None

# --------------------------------------------------------------------------
# 4. REPORT DATABASE SCHEMAS
# --------------------------------------------------------------------------
class RoadReport(BaseModel):
    id: Optional[str] = None
    lat: float
    lng: float
    severity: str
    description: Optional[str] = ""
    status: Optional[str] = "Pending"
    timestamp: Optional[str] = None

pothole_db: List[dict] = [
    {"id": "RG-101", "lat": 37.7755, "lng": -122.4180, "severity": "Severe", "status": "Pending", "timestamp": "2026-03-29 10:14"},
    {"id": "RG-102", "lat": 37.7730, "lng": -122.4210, "severity": "Moderate", "status": "Pending", "timestamp": "2026-03-29 11:30"}
]

# --------------------------------------------------------------------------
# 5. API ENDPOINTS
# --------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "RoadGuard & BharatCode Backend",
        "bharatcode_endpoint": BHARATCODE_BASE_URL
    }

@app.post("/api/v1/detect")
async def detect_potholes(file: UploadFile = File(...)):
    """Receives image frame, runs YOLO detection, and returns bounding boxes & severity."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

    detections = []
    if cv_model is not None:
        results = cv_model(img)
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > 0.40:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    severity = "Severe" if area > 25000 else ("Moderate" if area > 10000 else "Low")
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(conf, 2),
                        "severity": severity
                    })

    return {
        "success": True,
        "count": len(detections),
        "detections": detections
    }

@app.post("/api/v1/bharatcode/responses")
async def call_bharatcode_llm(req: BharatCodeRequest):
    """
    Proxy endpoint matching your BharatCode cURL documentation:
    POST https://bharatcode.ai/api/model/v1/responses
    Headers: Authorization: Bearer $BHARATCODE_API_KEY
    """
    url = f"{BHARATCODE_BASE_URL}/responses"
    headers = {
        "Authorization": f"Bearer {BHARATCODE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": req.model,
        "input": req.input,
        "store": req.store,
        "stream": req.stream
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": True,
                    "status_code": response.status_code,
                    "detail": response.text
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to reach BharatCode API: {str(e)}")

@app.get("/api/v1/reports")
def get_reports():
    return {"reports": pothole_db, "total": len(pothole_db)}

@app.post("/api/v1/reports")
def create_report(report: RoadReport):
    new_report = report.dict()
    new_report["id"] = f"RG-{len(pothole_db) + 101}"
    pothole_db.insert(0, new_report)
    return {"success": True, "report": new_report}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
