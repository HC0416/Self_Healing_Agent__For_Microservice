import os
import sys
import time
import threading
from datetime import datetime
from collections import deque
from common.llm import LLM

try:
    import docker
except Exception:
    docker = None

try:
    import requests
except Exception:
    requests = None

# Let the user point at wherever dynamic_healer.py lives.
_healer_dir = os.getenv("HEALER_DIR")
if _healer_dir and _healer_dir not in sys.path:
    sys.path.insert(0, _healer_dir)

# Recovery actions execute automatically when an anomaly passes cooldown checks.
# Set this to False only when you intentionally want observation-only mode.
AUTO_EXECUTE_RECOVERY = True

# Symptom features (from dynamic_healer.FEATURE_NAMES) used to label a fault
# type for the UI. is_anomaly / anomaly_score / error_ratio are excluded since
# they aren't specific symptoms.
SYMPTOM_FEATURES = [
    "service_crash", "slow_response", "overload_timeout",
    "network_delay", "db_error", "logic_error", "async_error",
]


class SystemMonitor:
    def __init__(self):
        from UI.backend import dynamic_healer as dh          # heavy import (torch, models)
        self.dh = dh
        print("Loading models for live dashboard (real SkyWalking healer)…")
        self.gnn, self.embedder, self.a3, self.actions, self.llm, self.tokenizer = dh.load_models()
        self.mon = dh.ServiceMonitor()        # the real baseline/cooldown logic

        self.recovery_history = deque(maxlen=50)
        self.recent_traces    = deque(maxlen=100)
        self.mttr_history     = []
        self.detection_latency_history = []
        self.action_counts    = {a: 0 for a in self.actions}
        self.total_recoveries = 0
        self.successful       = 0
        self.service_health   = {}            # name -> {name,status,score}
        self.target           = os.getenv("TARGET_SERVICE")
        self.skywalking_status = "online"
        self._lock = threading.RLock()
        self._active_recoveries = set()
        self._active_recovery_services = set()
        self._handled_crashed_services = set()

        # Ignore high-scoring startup/incomplete traces before allowing
        # automatic recovery actions.
        self.STARTUP_GRACE_SECONDS = float(
            os.getenv("STARTUP_GRACE_SECONDS", "60")
        )

        # Reuse one Docker client instead of reconnecting for every scan.
        self._docker_client = None

        print(
            "Real monitor ready. Mode: "
            f"{'AUTO EXECUTE' if AUTO_EXECUTE_RECOVERY else 'OBSERVE'}"
        )

    # ── helpers ────────────────────────────────────────────────
    def _infer_fault_type(self, feats):
        if feats.get("db_error", 0) > 0:
            return "db_error"
        if feats.get("service_crash", 0) > 0:
            return "service_crash"
        if feats.get("overload_timeout", 0) > 0:
            return "overload_timeout"
        if feats.get("slow_response", 0) > 0:
            return "slow_response"
        if feats.get("network_delay", 0) > 0:
            return "network_delay"
        if feats.get("logic_error", 0) > 0:
            return "logic_error"
        if feats.get("async_error", 0) > 0:
            return "async_error"
        return "anomaly"

    def _svc(self, name):
        return self.service_health.setdefault(
            name, {
                "name": name,
                "status": "healthy",
                "score": 0.001,
                "container_status": "unknown",
                "last_seen": None,
            }
        )

    def _skywalking_reachable(self):
        """Return True if the SkyWalking GraphQL endpoint is reachable.

        This is separate from get_all_services(), because dynamic_healer.get_all_services()
        catches exceptions and returns {}, so the dashboard would otherwise keep showing
        the last cached state when SkyWalking is down.
        """
        if requests is None:
            return True
        try:
            # The query does not need to be semantically important; if SkyWalking
            # replies with HTTP 200/400 and a GraphQL error, the server is still alive.
            resp = requests.post(
                self.dh.SW_GRAPHQL,
                json={"query": "{ version }"},
                headers={"Content-Type": "application/json"},
                timeout=3,
            )
            return resp.status_code < 500
        except Exception:
            return False

    def _get_docker_client(self):
        """Return a cached Docker client, or None when unavailable."""
        if docker is None:
            return None

        if self._docker_client is not None:
            return self._docker_client

        try:
            client = docker.from_env()
            client.ping()
            self._docker_client = client
            return client
        except Exception:
            self._docker_client = None
            return None

    def _docker_container_name(self, service):
        return f"train-ticket-{service}-1"

    def _docker_container_status(self, service):
        """Return Docker container status for a Train-Ticket service."""
        client = self._get_docker_client()

        if client is None:
            return "unknown"

        try:
            container = client.containers.get(
                self._docker_container_name(service)
            )
            container.reload()
            return container.status or "unknown"
        except Exception:
            return "missing"

    def _container_uptime_seconds(self, service):
        """Return Docker container uptime in seconds, or None."""
        client = self._get_docker_client()

        if client is None:
            return None

        try:
            container = client.containers.get(
                self._docker_container_name(service)
            )
            container.reload()

            state = container.attrs.get("State", {})
            status = state.get("Status") or container.status

            if status != "running":
                return 0.0

            started_at = state.get("StartedAt")

            if not started_at:
                return None

            value = str(started_at).strip()

            if value.endswith("Z"):
                value = value[:-1] + "+00:00"

            # Docker timestamps may have nanoseconds; Python datetime accepts
            # microseconds, so trim the fractional component to six digits.
            if "." in value:
                prefix, remainder = value.split(".", 1)

                if "+" in remainder:
                    fraction, offset = remainder.split("+", 1)
                    value = prefix + "." + fraction[:6] + "+" + offset
                elif "-" in remainder:
                    fraction, offset = remainder.split("-", 1)
                    value = prefix + "." + fraction[:6] + "-" + offset
                else:
                    value = prefix + "." + remainder[:6]

            started_dt = datetime.fromisoformat(value)
            now = datetime.now(started_dt.tzinfo)

            return max(
                0.0,
                (now - started_dt).total_seconds(),
            )

        except Exception:
            return None

    def _is_startup_trace(
        self,
        service,
        trace_duration,
        root_cause=None,
    ):
        """Return whether a trace should be treated as startup/incomplete."""
        uptime = self._container_uptime_seconds(service)

        if (
            uptime is not None
            and uptime < self.STARTUP_GRACE_SECONDS
        ):
            return (
                True,
                (
                    "container_startup_grace "
                    f"({uptime:.1f}s/"
                    f"{self.STARTUP_GRACE_SECONDS:.0f}s)"
                ),
                uptime,
            )

        try:
            duration = (
                int(trace_duration)
                if trace_duration is not None
                else None
            )
        except (TypeError, ValueError):
            duration = None

        root_text = str(root_cause or "").lower()

        is_connection_cleanup = any(
            keyword in root_text
            for keyword in (
                "connection/close",
                "hikaricp/connection/close",
                "mysql/jdbc/connection/close",
                "jdbc/connection/close",
            )
        )

        if (
            duration is not None
            and 0 < duration <= 5
            and is_connection_cleanup
        ):
            return (
                True,
                "short_connection_cleanup_trace",
                uptime,
            )

        if duration is not None and duration <= 0:
            return (
                True,
                "zero_duration_incomplete_trace",
                uptime,
            )

        return False, None, uptime

    def _refresh_container_statuses(self, service_names=None):
        """Update service_health using real Docker state.

        SkyWalking can still return services that existed recently, so Docker is the
        source of truth for whether the service container is currently running.
        """
        names = service_names or list(self.service_health.keys())
        for name in names:
            svc = self._svc(name)
            cstatus = self._docker_container_status(name)
            svc["container_status"] = cstatus
            if cstatus in ("exited", "dead", "created", "missing"):
                svc["status"] = "offline"
            elif cstatus in ("restarting", "paused"):
                svc["status"] = "warning"


    def _record_trace(self, *, service, trace_id, score, root_cause,duration_ms=None, threshold=None, status="NORMAL",fault_type=None, action_taken=None):    
        """Store every new SkyWalking trace for the trace dashboard.

        This is separate from recovery_history. recovery_history only stores
        anomaly/action events, while recent_traces stores normal + anomaly
        traces so the UI can show live trace flow.
        """
        self.recent_traces.appendleft({
            "type": "trace",
            "timestamp": datetime.now().isoformat(),
            "service": service,
            "trace_id": trace_id,
            "score": round(float(score), 6),
            "threshold": round(float(threshold), 6) if threshold is not None else None,
            "root_cause": root_cause or "unknown",
            "duration_ms": int(duration_ms) if duration_ms is not None else None,
            "status": status,
            "fault_type": fault_type or ("normal" if status == "NORMAL" else "—"),
            "action_taken": action_taken or "—",
        })

    def _find_trace(self, trace_id):
        for trace in self.recent_traces:
            if trace.get("trace_id") == trace_id:
                return trace
        return None

    def _find_recovery_event(self, trace_id):
        for event in self.recovery_history:
            if event.get("trace_id") == trace_id:
                return event
        return None

    def _start_recovery(
        self,
        *,
        service,
        trace_id,
        score,
        threshold,
        root_cause,
        fault_type,
        action,
        confidence,
        reason,
        execution_context,
        detection_latency_seconds=None,
    ):
        """Create a RECOVERING event immediately and run recovery in background."""
        if detection_latency_seconds is None:
            detection_latency_seconds = execution_context.get(
                "detection_latency_seconds"
            )

        event = {
            "trace_id": trace_id,
            "type": "recovery",
            "timestamp": datetime.now().isoformat(),
            "service": service,
            "score": round(float(score), 4),
            "fault_type": fault_type,
            "action": action,
            "rl_action": action,
            "reason": reason,
            "root_cause": root_cause,
            "recovery_status": "recovering",
            "executed": None,
            "verified": None,
            "recovered": None,
            "mttr": None,
            "attempt_duration": None,
            "recovery_message": "Recovery action is currently executing.",
            "recovery_details": {},
            "confidence": round(float(confidence), 1),
            "detection_latency_seconds": detection_latency_seconds,
        }

        with self._lock:
            self.recovery_history.appendleft(event)
            self._svc(service)["status"] = "recovering"
            self._active_recoveries.add(trace_id)
            self._active_recovery_services.add(service)

        worker = threading.Thread(
            target=self._complete_recovery,
            kwargs={
                "service": service,
                "trace_id": trace_id,
                "fault_type": fault_type,
                "action": action,
                "execution_context": execution_context,
            },
            daemon=True,
            name=f"recovery-{service}-{trace_id[-8:]}",
        )
        worker.start()

    def _complete_recovery(self, *, service, trace_id, fault_type, action, execution_context):
        """Execute, verify, and update the existing recovery event."""
        dh = self.dh

        try:
            if AUTO_EXECUTE_RECOVERY:
                if fault_type == "db_error":
                    recovery_result = (
                        dh.recovery_executor
                        .restart_database(service)
                    )

                    recovery_result.setdefault("rl_action", action)
                    recovery_result.setdefault(
                        "recommended_action",
                        action,
                    )
                    recovery_result.setdefault(
                        "executed_action",
                        recovery_result.get("action", action),
                    )
                    recovery_result.setdefault(
                        "action_adapted",
                        False,
                    )
                    recovery_result.setdefault(
                        "adaptation_reason",
                        "Dedicated database recovery path",
                    )
                else:
                    recovery_result = (
                        dh.recovery_executor.execute(
                            action=action,
                            service=service,
                            approved=False,
                            context=execution_context,
                        )
                    )
            else:
                recovery_result = {
                    "action": action,
                    "rl_action": action,
                    "recommended_action": action,
                    "executed_action": action,
                    "action_adapted": False,
                    "adaptation_reason": (
                        "Observe mode; no action executed"
                    ),
                    "service": service,
                    "executed": False,
                    "verified": False,
                    "recovered": False,
                    "manual_required": False,
                    "mttr": None,
                    "attempt_duration": 0.0,
                    "message": (
                        "Observe mode: action selected but not executed"
                    ),
                    "details": {
                        "observe_mode": True,
                        **execution_context,
                    },
                }

        except Exception as exc:
            recovery_result = {
                "action": action,
                "rl_action": action,
                "recommended_action": action,
                "executed_action": action,
                "action_adapted": False,
                "adaptation_reason": (
                    "Recovery execution raised an exception"
                ),
                "service": service,
                "executed": False,
                "verified": False,
                "recovered": False,
                "manual_required": False,
                "mttr": None,
                "attempt_duration": 0.0,
                "message": f"Recovery execution failed: {exc}",
                "details": {
                    "exception": repr(exc),
                    **execution_context,
                },
            }

        recommended_action = recovery_result.get(
            "recommended_action",
            recovery_result.get("rl_action", action),
        )

        executed_action = recovery_result.get(
            "executed_action",
            recovery_result.get("action", action),
        )

        action_adapted = bool(
            recovery_result.get("action_adapted", False)
        )
        adaptation_reason = recovery_result.get(
            "adaptation_reason"
        )

        details = recovery_result.get("details", {}) or {}

        manual_required = bool(
            recovery_result.get("manual_required")
            or details.get("manual_required")
            or details.get("approval_required")
        )

        executed = bool(recovery_result.get("executed"))
        verified = bool(recovery_result.get("verified"))
        recovered = bool(recovery_result.get("recovered", verified))

        raw_mttr = recovery_result.get("mttr")
        mttr = (
            float(raw_mttr)
            if raw_mttr is not None
            else None
        )

        raw_attempt = recovery_result.get("attempt_duration")
        attempt_duration = (
            float(raw_attempt)
            if raw_attempt is not None
            else 0.0
        )

        if manual_required:
            recovery_status = "manual_required"
        elif verified:
            recovery_status = "recovered"
        else:
            recovery_status = "failed"

        with self._lock:
            trace = self._find_trace(trace_id)
            event = self._find_recovery_event(trace_id)

            if trace is not None:
                trace["rl_action"] = recommended_action
                trace["action_taken"] = executed_action
                trace["action_adapted"] = action_adapted
                trace["adaptation_reason"] = adaptation_reason
                trace["recovery_status"] = recovery_status
                trace["manual_required"] = manual_required
                trace["detection_latency_seconds"] = (
                    event.get("detection_latency_seconds")
                    if event is not None
                    else trace.get("detection_latency_seconds")
                )

            if event is not None:
                event.update({
                    "rl_action": recommended_action,
                    "recommended_action": recommended_action,
                    "action": executed_action,
                    "executed_action": executed_action,
                    "action_adapted": action_adapted,
                    "adaptation_reason": adaptation_reason,
                    "executed": executed,
                    "verified": verified,
                    "recovered": recovered,
                    "manual_required": manual_required,
                    "detection_latency_seconds": event.get(
                        "detection_latency_seconds"
                    ),
                    "mttr": (
                        None if manual_required else mttr
                    ),
                    "attempt_duration": attempt_duration,
                    "recovery_status": recovery_status,
                    "recovery_message": recovery_result.get(
                        "message",
                        "",
                    ),
                    "recovery_details": details,
                    "target": (
                        details.get("target")
                        or details.get("backup")
                        or details.get("container")
                    ),
                    "verification_method": details.get(
                        "verification_method",
                        (
                            "Operator approval"
                            if manual_required
                            else "Docker / application health verification"
                        ),
                    ),
                    "verification_status": details.get(
                        "verification_status",
                        (
                            "Manual rollback required"
                            if manual_required
                            else ("Passed" if verified else "Failed")
                        ),
                    ),
                })

            self.total_recoveries += 1
            self.action_counts[executed_action] = (
                self.action_counts.get(executed_action, 0)
                + 1
            )

            if verified:
                self.successful += 1
                self._svc(service)["status"] = "healthy"

                if mttr is not None:
                    self.mttr_history.append(mttr)
                    self.mttr_history = self.mttr_history[-30:]

            elif manual_required:
                self._svc(service)["status"] = "manual_required"

            else:
                self._svc(service)["status"] = "warning"

            self._active_recoveries.discard(trace_id)
            self._active_recovery_services.discard(service)

        # A manual rollback is not healed yet. Do not suppress future anomalies
        # as if the system had recovered.
        if not manual_required:
            self.mon.mark_healed(service, trace_id)

    # ── main cycle ─────────────────────────────────────────────
    def run_once(self):
        dh = self.dh

        # 1) Check SkyWalking liveness first. If SkyWalking is down, do not keep
        # showing old traces as if they are live.
        if not self._skywalking_reachable():
            print("SkyWalking GraphQL is offline")
            self.skywalking_status = "offline"
            self.recent_traces.clear()
            self.recovery_history.clear()
            for svc in self.service_health.values():
                svc["status"] = "stale"
                svc["container_status"] = "unknown"
            return self._build_update()

        self.skywalking_status = "online"

        try:
            services = dh.get_all_services()

            INFRA_KEYWORDS = [
                "mysql",
                "mongo",
                "mongodb",
                "redis",
                "rabbitmq",
                "kafka",
                "nacos",
                "skywalking",
                "elasticsearch",
                ":3306",
                ":27017",
            ]

            services = {
                name: sid
                for name, sid in services.items()
                if not any(k in name.lower() for k in INFRA_KEYWORDS)
            }

        except Exception as e:
            print(f"get_all_services error: {e}")
            self.skywalking_status = "offline"
            self.recent_traces.clear()
            self.recovery_history.clear()
            for svc in self.service_health.values():
                svc["status"] = "stale"
            return self._build_update()

        if self.target:
            services = {k: v for k, v in services.items() if self.target in k}

        # If SkyWalking is reachable but no matching services are returned, keep the
        # dashboard honest by marking previously-known services as offline/stale.
        if not services:
            for svc in self.service_health.values():
                svc["status"] = "offline"
            self._refresh_container_statuses()
            return self._build_update()

        # SkyWalking returns historical services, so refresh Docker state before
        # processing traces. Stopped containers should not appear healthy.
        self._refresh_container_statuses(services.keys())

        current_service_names = set(services.keys())
        for known_name in list(self.service_health.keys()):
            if known_name not in current_service_names:
                self.service_health[known_name]["status"] = "offline"

        for name, sid in services.items():
            svc = self._svc(name)
            svc["last_seen"] = datetime.now().isoformat()

            container_status = svc.get("container_status")

            if container_status not in ("running","unknown",):
                svc["status"] = "offline"

                if container_status in {"exited","dead","created","restarting",}:
                    self._handle_container_crash(name,container_status,)

                continue

            try:
                trace = dh.get_latest_trace(sid)
                if not trace:
                    continue
                tid = trace["traceIds"][0]
                trace_duration = trace.get("duration")

                # only act on genuinely new traces (safe under fast polling)
                if self.mon.last_trace.get(name) == tid:
                    continue
                self.mon.last_trace[name] = tid

                # Live detection latency starts when the monitor accepts a new
                # SkyWalking trace for analysis. It ends when the anomaly
                # decision and RL action are ready.
                detection_started = time.perf_counter()

                spans = dh.get_spans(tid)
                if not spans:
                    continue
                score, root_cause = dh.score_spans(spans, self.gnn, self.embedder)
                if score is None:
                    continue

                self._svc(name)["score"] = round(float(score), 4)

                # Startup and incomplete traces can produce very large GNN
                # scores even when the service is healthy. Display them as
                # WARMUP, but do not update the baseline or execute recovery.
                is_startup, startup_reason, uptime = (
                    self._is_startup_trace(
                        name,
                        trace_duration,
                        root_cause,
                    )
                )

                if is_startup:
                    startup_threshold = None

                    if self.mon.baseline_ready(name):
                        try:
                            startup_threshold = self.mon.get_threshold(name)
                        except Exception:
                            startup_threshold = None

                    self._svc(name)["status"] = "warming_up"

                    self._record_trace(
                        service=name,
                        trace_id=tid,
                        score=score,
                        root_cause=root_cause,
                        duration_ms=trace_duration,
                        threshold=startup_threshold,
                        status="WARMUP",
                        fault_type="startup",
                        action_taken="—",
                    )

                    print(
                        f"[{name}] ignored startup/incomplete trace: "
                        f"{startup_reason}; "
                        f"score={float(score):.6f}; "
                        f"duration={trace_duration}"
                    )

                    continue

                # Default trace status is updated below once baseline/threshold is known.
                trace_recorded = False

                # polluted-baseline reset
                if (self.mon.baselines.get(name, 0.0) > self.mon.POLLUTED_BASELINE_MAX
                        and score < self.mon.CLEAN_SCORE_MAX):
                    self.mon.baselines[name] = score
                    self.mon.baseline_counts[name] = 1
                    self.mon.warmup_scores[name] = [score]
                    self.mon.thresholds.pop(name, None)
                    if hasattr(self.mon, "save_baselines"):
                        self.mon.save_baselines()
                    self._svc(name)["status"] = "healthy"
                    self._record_trace(service=name, trace_id=tid, score=score, root_cause=root_cause, duration_ms=trace_duration, threshold=None, status="BASELINE_RESET")
                    trace_recorded = True
                    continue

                # warm-up
                if not self.mon.baseline_ready(name):
                    # Do not allow obvious fault-level scores into the baseline.
                    # Record them as WARNING in the UI instead of incorrectly
                    # labelling them as WARMUP.
                    if score > self.mon.MAX_BASELINE_SCORE:
                        self._svc(name)["status"] = "warning"
                        self._record_trace(
                            service=name,
                            trace_id=tid,
                            score=score,
                            root_cause=root_cause,
                            duration_ms=trace_duration,
                            threshold=None,
                            status="WARNING",
                            fault_type="baseline_skip",
                            action_taken="—",
                        )
                        trace_recorded = True
                        continue

                    self.mon.update_baseline(name, score)
                    count = self.mon.baseline_counts.get(name, 0)

                    if count >= self.mon.BASELINE_WARMUP_TRACES:
                        self.mon.finalize_threshold(name)
                        threshold = self.mon.thresholds.get(name)
                        self._svc(name)["status"] = "healthy"
                        self._record_trace(
                            service=name,
                            trace_id=tid,
                            score=score,
                            root_cause=root_cause,
                            duration_ms=trace_duration,
                            threshold=threshold,
                            status="WARMUP",
                            fault_type="—",
                            action_taken="—",
                        )
                        print(
                            f"[{name}] baseline ready: "
                            f"baseline={self.mon.baselines[name]:.6f}, "
                            f"threshold={threshold:.6f}"
                        )
                    else:
                        self._svc(name)["status"] = "healthy"
                        self._record_trace(
                            service=name,
                            trace_id=tid,
                            score=score,
                            root_cause=root_cause,
                            duration_ms=trace_duration,
                            threshold=None,
                            status="WARMUP",
                            fault_type="—",
                            action_taken="—",
                        )

                    trace_recorded = True
                    continue

                normal = self.mon.baselines.get(name, score)
                thr = self.mon.get_threshold(name)

                if score < thr:
                    self.mon.update_baseline(name, score)
                    self._svc(name)["status"] = "healthy"
                    self._record_trace(service=name, trace_id=tid, score=score, root_cause=root_cause, duration_ms=trace_duration, threshold=thr, status="NORMAL")
                    trace_recorded = True
                    continue

                # Record anomaly, but never use it to update the healthy baseline.
                self._record_trace(
                    service=name,
                    trace_id=tid,
                    score=score,
                    root_cause=root_cause,
                    duration_ms=trace_duration,
                    threshold=thr,
                    status="ANOMALY",
                )
                trace_recorded = True

                # Diagnose before cooldown/repetition suppression so the UI is
                # updated consistently for both primary and backup services.
                state = dh.build_state_from_trace(
                    score,
                    spans,
                    name,
                )

                feats = dict(
                    zip(
                        dh.FEATURE_NAMES,
                        state,
                    )
                )

                fault_type = self._infer_fault_type(feats)
                trace_event = self._find_trace(tid)

                if fault_type == "anomaly":
                    self._svc(name)["status"] = "warning"

                    if trace_event is not None:
                        trace_event["status"] = "WARNING"
                        trace_event["fault_type"] = "unknown"
                        trace_event["action_taken"] = "—"

                    print(
                        f"[{name}] anomaly exceeded threshold, "
                        "but no specific fault symptom was identified; "
                        "automatic recovery suppressed"
                    )

                    continue

                if self.mon.recently_healed(name, tid):
                    self._svc(name)["status"] = "warning"

                    if trace_event is not None:
                        trace_event["status"] = "WARNING"
                        trace_event["fault_type"] = fault_type
                        trace_event["action_taken"] = "—"

                    continue

                if self.mon.repeated_anomaly(name, score):
                    self._svc(name)["status"] = "warning"

                    if trace_event is not None:
                        trace_event["status"] = "WARNING"
                        trace_event["fault_type"] = fault_type
                        trace_event["action_taken"] = "—"

                    continue

                self._svc(name)["status"] = "fault"

                action, confidence, probs = dh.select_action(
                    state,
                    self.a3,
                    self.actions,
                )

                # Detection latency is measured fully in memory:
                # new trace accepted -> spans fetched -> GNN scored ->
                # fault diagnosed -> RL action selected.
                detection_latency = round(
                    time.perf_counter() - detection_started,
                    3,
                )

                with self._lock:
                    self.detection_latency_history.append(
                        detection_latency
                    )
                    self.detection_latency_history = (
                        self.detection_latency_history[-30:]
                    )

                print(
                    f"[Detection] {name} trace={tid} "
                    f"latency={detection_latency:.3f}s"
                )

                if trace_event is not None:
                    trace_event["fault_type"] = fault_type
                    trace_event["action_taken"] = action

                reason = None
                try:
                    llm = getattr(self, "llm", None)
                    tokenizer = getattr(self, "tokenizer", None)

                    if llm is not None and tokenizer is not None:
                        reason = LLM.ask_llm(
                            action,
                            state,
                            score,
                            normal,
                            root_cause,
                            name,
                            llm,
                            tokenizer,
                        )
                except Exception:
                    reason = None
                if not reason:
                    reason = "ERROR IN REASONING"

                execution_context = {
                    "trace_id": tid,
                    "fault_type": fault_type,
                    "root_cause": root_cause,
                    "score": float(score),
                    "threshold": float(thr),
                    "rl_action": action,
                    "detection_latency_seconds": detection_latency,
                }

                # Add the event now, then execute recovery in a background thread.
                self._start_recovery(
                    service=name,
                    trace_id=tid,
                    score=score,
                    threshold=thr,
                    root_cause=root_cause,
                    fault_type=fault_type,
                    action=action,
                    confidence=confidence,
                    reason=reason,
                    execution_context=execution_context,
                    detection_latency_seconds=detection_latency,
                )

                # The WebSocket receives this cycle immediately with
                # recovery_status="recovering".
                continue


            except Exception as e:
                print(f"scan error on {name}: {e}")
                continue

        return self._build_update()

    # ── output shaping (matches mock_monitor) ──────────────────
    def get_current_state(self):
        return {"type": "full_state", **self._build_update()}

    def _build_update(self):
        with self._lock:
            total = max(self.total_recoveries, 1)
            succ = round(self.successful / total * 100, 1)
            recent = list(self.mttr_history[-15:])
            avg = round(sum(recent) / len(recent), 2) if recent else 0.0

            recent_detection = list(
                self.detection_latency_history[-15:]
            )
            avg_detection = (
                round(
                    sum(recent_detection) / len(recent_detection),
                    2,
                )
                if recent_detection
                else None
            )

            return {
                "services": [dict(service) for service in self.service_health.values()],
                "action_counts": dict(self.action_counts),
                "recent_actions": [dict(event) for event in list(self.recovery_history)[:10]],
                "recent_traces": [dict(trace) for trace in list(self.recent_traces)[:50]],
                "mttr_history": recent,
                "metrics": {
                    "total_recoveries": self.total_recoveries,
                    "success_rate": succ,
                    "avg_mttr": avg,
                    "avg_detection_latency": avg_detection,
                    "detection_latency_samples": len(
                        recent_detection
                    ),
                    "false_alarms": 0,
                },
                "skywalking_status": self.skywalking_status,
            }

    def get_metrics(self):
        return self._build_update()["metrics"]

    def get_traces(self):
        return list(self.recent_traces)

    def _build_service_crash_state(self):
        return [
            1.0,  # anomaly_score
            1.0,  # service_crash
            0.0,  # slow_response
            0.0,  # overload_timeout
            0.0,  # network_delay
            1.0,  # is_anomaly
            0.0,  # db_error
            0.0,  # logic_error
            0.0,  # async_error
            1.0,  # complete failure
        ]
    
    def _handle_container_crash(self,service: str,container_status: str,):
        """
        Handle a complete service crash detected through Docker.

        No SkyWalking trace is expected because the service is stopped.
        The real RL policy still selects the recovery action.
        """

        # Avoid repeatedly handling the same stopped service while
        # a recovery is already running.
        if service in self._active_recovery_services:
            return

        # Docker-crash detection latency starts when the stopped container
        # is accepted for analysis and ends when the RL action is selected.
        detection_started = time.perf_counter()

        state = self._build_service_crash_state()

        action, confidence, probs = self.dh.select_action(
            state,
            self.a3,
            self.actions,
        )

        detection_latency = round(
            time.perf_counter() - detection_started,
            3,
        )

        with self._lock:
            self.detection_latency_history.append(
                detection_latency
            )
            self.detection_latency_history = (
                self.detection_latency_history[-30:]
            )

        print(
            f"[Detection/Docker] {service} "
            f"latency={detection_latency:.3f}s"
        )

        fault_type = "service_crash"

        print(
            f"[Docker crash] {service}: "
            f"status={container_status}, "
            f"RL action={action}, "
            f"confidence={confidence:.1f}%"
        )

        # A synthetic event ID is needed because there is no trace ID.
        event_id = (
            f"docker-crash-{service}-"
            f"{int(time.time() * 1000)}"
        )

        execution_context = {
            "trace_id": event_id,
            "fault_type": fault_type,
            "container_status": container_status,
            "root_cause": "Docker container unavailable",
            "score": 1.0,
            "threshold": 0.0,
            "rl_action": action,
            "detection_source": "docker",
            "detection_latency_seconds": detection_latency,
        }

        self._record_trace(
            service=service,
            trace_id=event_id,
            score=1.0,
            root_cause="Docker container unavailable",
            duration_ms=None,
            threshold=0.0,
            status="ANOMALY",
            fault_type=fault_type,
            action_taken=action,
        )

        self._start_recovery(
            service=service,
            trace_id=event_id,
            score=1.0,
            threshold=0.0,
            root_cause="Docker container unavailable",
            fault_type=fault_type,
            action=action,
            confidence=confidence,
            reason=(
                "Docker reported that the service container "
                "was not running. The RL policy selected "
                f"{action}."
            ),
            execution_context=execution_context,
            detection_latency_seconds=detection_latency,
        )

def main() -> None:
    """Run the real monitor continuously from the command line."""
    poll_seconds = float(os.getenv("MONITOR_POLL_SECONDS", "3"))
    monitor = SystemMonitor()

    try:
        while True:
            cycle_started = time.time()

            try:
                update = monitor.run_once()
                metrics = update.get("metrics", {})
                services = update.get("services", [])

                healthy = sum(
                    1 for service in services
                    if service.get("status") == "healthy"
                )
                faults = sum(
                    1 for service in services
                    if service.get("status") in {
                        "fault",
                        "recovering",
                        "warning",
                        "offline",
                    }
                )

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"scan complete | services={len(services)} "
                    f"healthy={healthy} attention={faults} "
                    f"recoveries={metrics.get('total_recoveries', 0)}"
                )

            except Exception as exc:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"monitor cycle failed: {exc}"
                )

            elapsed = time.time() - cycle_started
            time.sleep(max(0.0, poll_seconds - elapsed))

    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")


if __name__ == "__main__":
    main()