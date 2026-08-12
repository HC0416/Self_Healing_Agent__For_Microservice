import os
import random
import threading
import uuid
from collections import deque
from datetime import datetime
from typing import Any

from UI.backend.recovery_executor import RecoveryExecutor

MONITORED_SERVICES = [
    "ts-auth-service", "ts-order-service", "ts-payment-service",
    "ts-execute-service", "ts-station-service", "ts-travel-service",
    "ts-cancel-service", "ts-preserve-service", "ts-basic-service",
]

FAULT_CHANCE = 0.0
FAULT_COOLDOWN_TICKS = int(os.getenv("MOCK_FAULT_COOLDOWN_TICKS", "4"))

# FAULT_SCENARIOS = [{
#     "fault_type": "service_crash",
#     "rl_action": "RESTART",
#     "score": 0.891,
#     "threshold": 0.180,
#     "root_cause": "ts-auth-service container unavailable",
#     "confidence": 98.4,
#     "reason": (
#         "Detected fault type: service_crash. "
#         "The simulated RL policy selected RESTART."
#     ),
# }]

# FAULT_SCENARIOS = [
#     {
#         "fault_type": "network_delay",
#         "rl_action": "REROUTE",
#         "score": 0.621,
#         "threshold": 0.180,
#         "root_cause": (
#             "degraded communication path "
#             "on primary auth instance"
#         ),
#         "confidence": 89.8,
#         "reason": (
#             "Detected fault type: network_delay. "
#             "The simulated RL policy selected REROUTE."
#         ),
#     },
# ]

# FAULT_SCENARIOS = [{
#     "fault_type": "overload_timeout",
#     "rl_action": "SCALE_UP",
#     "score": 0.734,
#     "threshold": 0.180,
#     "root_cause": "CPU or memory resource pressure",
#     "confidence": 94.2,
#     "reason": (
#         "Detected fault type: overload_timeout. "
#         "The simulated RL policy selected SCALE_UP."
#     ),
# }]

FAULT_SCENARIOS = [
    {
        "fault_type": "logic_error",
        "rl_action": "ROLLBACK",
        "score": 0.654,
        "threshold": 0.180,
        "root_cause": (
            "deterministic application logic regression"
        ),
        "confidence": 93.4,
        "reason": (
            "Detected fault type: logic_error. "
            "The simulated RL policy selected ROLLBACK."
        ),
    },

    # {
    #     "fault_type": "service_crash",
    #     "rl_action": "RESTART",
    #     "score": 0.75,
    #     "threshold": 0.18,
    #     "root_cause": "primary service process unavailable",
    #     "confidence": 95.0,
    #     "reason": (
    #         "Detected fault type: service_crash. "
    #         "The simulated RL policy selected RESTART."
    #     ),
    # }
]


class SystemMonitor:
    def __init__(self, *args: Any, **kwargs: Any):
        self.recovery_executor = RecoveryExecutor()
        self.recovery_history = deque(maxlen=50)
        self.recent_traces = deque(maxlen=100)
        self.mttr_history: list[float] = []
        self.action_counts = {a: 0 for a in [
            "RESTART", "RESTART_DATABASE", "SCALE_UP",
            "CIRCUIT_BREAK", "REROUTE", "ROLLBACK",
        ]}
        self.total_recoveries = 0
        self.successful = 0
        self.service_health = {
            service: {
                "name": service,
                "status": "healthy",
                "score": round(random.uniform(0.001, 0.005), 4),
                "container_status": "mock",
                "last_seen": datetime.now().isoformat(),
            }
            for service in MONITORED_SERVICES
        }
        self._tick = 0
        self._last_fault_tick = -FAULT_COOLDOWN_TICKS
        self._lock = threading.RLock()
        self.skywalking_status = "mock"
        print("Mock monitor ready — simulated GNN/RL with REAL RecoveryExecutor")

    def _jitter_scores(self):
        with self._lock:
            for state in self.service_health.values():
                if state["status"] == "healthy":
                    current = float(state["score"]) + random.uniform(-0.0008, 0.0008)
                    state["score"] = round(min(max(current, 0.001), 0.006), 4)
                    state["last_seen"] = datetime.now().isoformat()

    def run_once(self):
        self._tick += 1
        # self._inject_fault()
        self._jitter_scores()
        return self._build_update()

    def prepare_fault(self, fault_type=None, service=None):
        selected_service = service or "ts-auth-service"
        scenario = FAULT_SCENARIOS[0]
        trace_id = f"mock.{uuid.uuid4().hex}"
        now = datetime.now().isoformat()
        score = float(scenario["score"])
        threshold = float(scenario["threshold"])
        selected_fault = scenario["fault_type"]
        rl_action = scenario["rl_action"]

        trace = {
            "type": "trace",
            "timestamp": now,
            "service": selected_service,
            "trace_id": trace_id,
            "score": round(score, 6),
            "threshold": round(threshold, 6),
            "root_cause": scenario["root_cause"],
            "duration_ms": random.randint(100, 5000),
            "status": "ANOMALY",
            "fault_type": selected_fault,
            "action_taken": rl_action,
        }

        event = {
            "trace_id": trace_id,
            "type": "recovery",
            "timestamp": now,
            "service": selected_service,
            "score": round(score, 6),
            "fault_type": selected_fault,
            "action": rl_action,
            "rl_action": rl_action,
            "reason": scenario["reason"],
            "shap_table": self._mock_shap_table(selected_fault),
            "root_cause": scenario["root_cause"],
            "recovery_status": "recovering",
            "executed": None,
            "verified": None,
            "recovered": None,
            "mttr": None,
            "attempt_duration": None,
            "recovery_message": "Recovery action is currently executing.",
            "recovery_details": {},
            "confidence": float(scenario["confidence"]),
            "source": "mock",
        }

        with self._lock:
            self.service_health.setdefault(selected_service, {
                "name": selected_service,
                "status": "healthy",
                "score": 0.002,
                "container_status": "mock",
                "last_seen": now,
            })
            self.service_health[selected_service].update({
                "status": "recovering",
                "score": score,
                "last_seen": now,
            })
            self.recent_traces.appendleft(trace)
            self.recovery_history.appendleft(event)

        return {
            "trace_id": trace_id,
            "service": selected_service,
            "scenario": scenario,
            "context": {
                "trace_id": trace_id,
                "fault_type": selected_fault,
                "root_cause": scenario["root_cause"],
                "score": score,
                "threshold": threshold,
                "rl_action": rl_action,
                "source": "mock",
            },
        }

    def execute_prepared_fault(self, job):
        trace_id = job["trace_id"]
        service = job["service"]
        scenario = job["scenario"]
        context = job["context"]
        fault_type = scenario["fault_type"]
        rl_action = scenario["rl_action"]

        try:
            if fault_type == "db_error":
                result = self.recovery_executor.restart_database(service)
            else:
                result = self.recovery_executor.execute(
                    action=rl_action,
                    service=service,
                    approved=False,
                    context=context,
                )
        except Exception as exc:
            result = {
                "action": rl_action,
                "service": service,
                "executed": False,
                "verified": False,
                "recovered": False,
                "mttr": None,
                "attempt_duration": 0.0,
                "message": f"Recovery execution failed: {exc}",
                "details": {"exception": repr(exc), **context},
            }

        executed_action = result.get("action", rl_action)
        executed = bool(result.get("executed"))
        verified = bool(result.get("verified"))
        recovered = bool(result.get("recovered", verified))
        raw_mttr = result.get("mttr")
        mttr = float(raw_mttr) if raw_mttr is not None else None
        raw_attempt = result.get("attempt_duration")
        attempt_duration = float(raw_attempt) if raw_attempt is not None else 0.0

        with self._lock:
            trace = self._find_trace(trace_id)
            event = self._find_event(trace_id)

            if trace is not None:
                trace["action_taken"] = executed_action

            if event is not None:
                event.update({
                    "action": executed_action,
                    "executed": executed,
                    "verified": verified,
                    "recovered": recovered,
                    "mttr": mttr,
                    "attempt_duration": attempt_duration,
                    "recovery_status": "recovered" if verified else "failed",
                    "recovery_message": result.get("message", ""),
                    "recovery_details": result.get("details", {}),
                })

            self.total_recoveries += 1
            self.action_counts[executed_action] = self.action_counts.get(executed_action, 0) + 1

            if verified:
                self.successful += 1
                self.service_health[service]["status"] = "healthy"
                self.service_health[service]["score"] = round(random.uniform(0.001, 0.005), 4)
                if mttr is not None:
                    self.mttr_history.append(mttr)
                    self.mttr_history = self.mttr_history[-30:]
            else:
                self.service_health[service]["status"] = "warning"

            self.service_health[service]["last_seen"] = datetime.now().isoformat()
            return dict(event) if event else result

    def trigger_fault(self, fault_type=None, service=None):
        job = self.prepare_fault(fault_type, service)
        return self.execute_prepared_fault(job)

    def _find_trace(self, trace_id):
        return next((t for t in self.recent_traces if t.get("trace_id") == trace_id), None)

    def _find_event(self, trace_id):
        return next((e for e in self.recovery_history if e.get("trace_id") == trace_id), None)

    def _mock_shap_table(self, fault_type):
        feature = "service_crash" if fault_type == "db_error" else fault_type
        return (
            f"  {'Feature':20s} {'Value':8s} {'SHAP':10s} {'Impact'}\n"
            f"  {'─' * 55}\n"
            f"  {feature:20s} {1.0:8.3f} {0.75:+10.4f}  +███████████████"
        )

    def get_current_state(self):
        return {"type": "full_state", **self._build_update()}

    def _build_update(self):
        with self._lock:
            total = max(self.total_recoveries, 1)
            recent = list(self.mttr_history[-15:])
            return {
                "services": [dict(x) for x in self.service_health.values()],
                "action_counts": dict(self.action_counts),
                "recent_actions": [dict(x) for x in list(self.recovery_history)[:10]],
                "recent_traces": [dict(x) for x in list(self.recent_traces)[:50]],
                "mttr_history": recent,
                "metrics": {
                    "total_recoveries": self.total_recoveries,
                    "success_rate": round(self.successful / total * 100, 1),
                    "avg_mttr": round(sum(recent) / len(recent), 2) if recent else 0.0,
                    "false_alarms": 0,
                },
                "skywalking_status": self.skywalking_status,
            }

    def get_metrics(self):
        return self._build_update()["metrics"]

    def get_traces(self):
        with self._lock:
            return [dict(x) for x in self.recent_traces]