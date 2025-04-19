from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import subprocess
import uuid
import asyncio
import json

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active processes store
active_processes = {}

@app.post("/start-posture/{posture_name}")
async def start_posture(posture_name: str):
    process_id = str(uuid.uuid4())
    script_path = f"./posture_scripts/{posture_name}.py"
    
    # Start the posture script as subprocess
    process = subprocess.Popen(
        ["python", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    active_processes[process_id] = process
    return {"message": "Posture analysis started", "process_id": process_id}

@app.post("/stop-posture/{process_id}")
async def stop_posture(process_id: str):
    process = active_processes.get(process_id)
    if process:
        process.terminate()
        del active_processes[process_id]
        return {"message": "Posture analysis stopped"}
    return {"message": "Process not found"}

@app.websocket("/ws/{process_id}")
async def websocket_endpoint(websocket: WebSocket, process_id: str):
    await websocket.accept()
    try:
        while True:
            process = active_processes.get(process_id)
            if process:
                # Read output from subprocess
                output = process.stdout.readline()
                if output:
                    await websocket.send_text(json.dumps({
                        "type": "data",
                        "content": output.strip()
                    }))
                await asyncio.sleep(0.1)
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": "Process not found"
                }))
                break
    except WebSocketDisconnect:
        print("Client disconnected")