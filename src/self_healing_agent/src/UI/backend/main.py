import asyncio
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MONITOR_MODE: Literal["mock", "real"] = "real" # Change to "mock" for mock mode
POLL_INTERVAL = 2.0


def make_monitor():
    if MONITOR_MODE == "mock":
        from UI.backend.mock_monitor import SystemMonitor
        print("Using MOCK anomaly source with REAL recovery execution.")
        return SystemMonitor()
    
    from UI.backend.monitor import SystemMonitor
    print("Using REAL SkyWalking monitor.")
    return SystemMonitor()


class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, data: dict):
        for websocket in self.active.copy():
            try:
                await websocket.send_json(data)
            except Exception:
                self.disconnect(websocket)


manager = ConnectionManager()
monitor = make_monitor()
monitor_task: Optional[asyncio.Task] = None
recovery_tasks: set[asyncio.Task] = set()


async def monitor_loop():
    print(f"Monitor loop started in {MONITOR_MODE.upper()} mode with poll interval={POLL_INTERVAL}s")
    while True:
        try:
            update = await asyncio.to_thread(monitor.run_once)
            if update:
                await manager.broadcast(update)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"Monitor error: {exc}")
        await asyncio.sleep(POLL_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global monitor_task
    monitor_task = asyncio.create_task(monitor_loop())
    try:
        yield
    finally:
        if monitor_task:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        for task in list(recovery_tasks):
            task.cancel()


app = FastAPI(title="Sentinel API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json(monitor.get_current_state())
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        manager.disconnect(websocket)


@app.get("/api/status")
def get_status():
    return monitor.get_current_state()


@app.get("/api/history")
def get_history():
    return {"history": list(monitor.recovery_history)}


@app.get("/api/metrics")
def get_metrics():
    return monitor.get_metrics()


@app.get("/api/mode")
def get_mode():
    return {
        "mode": MONITOR_MODE,
        "poll_interval": POLL_INTERVAL,
        "real_recovery_enabled": True,
    }


class MockFaultRequest(BaseModel):
    fault_type: Optional[str] = None
    service: Optional[str] = None


async def run_mock_recovery(job: dict):
    try:
        await asyncio.to_thread(monitor.execute_prepared_fault, job)
    except Exception as exc:
        print(f"Mock recovery failed: {exc}")
    finally:
        await manager.broadcast(monitor.get_current_state())


@app.post("/api/mock/trigger")
async def trigger_mock_fault(request: MockFaultRequest):
    if MONITOR_MODE != "mock":
        raise HTTPException(status_code=400, detail="Mock mode only")

    prepare = getattr(monitor, "prepare_fault", None)
    execute = getattr(monitor, "execute_prepared_fault", None)

    if not callable(prepare) or not callable(execute):
        raise HTTPException(
            status_code=500,
            detail="Mock monitor requires prepare_fault() and execute_prepared_fault().",
        )

    job = prepare(request.fault_type, request.service)

    await manager.broadcast(monitor.get_current_state())

    task = asyncio.create_task(run_mock_recovery(job))
    recovery_tasks.add(task)
    task.add_done_callback(recovery_tasks.discard)

    return {
        "accepted": True,
        "trace_id": job["trace_id"],
        "service": job["service"],
        "fault_type": job["scenario"]["fault_type"],
        "action": job["scenario"]["rl_action"],
        "recovery_status": "recovering",
    }