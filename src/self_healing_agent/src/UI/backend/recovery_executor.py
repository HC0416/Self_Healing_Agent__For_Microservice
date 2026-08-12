"""Structured recovery action execution for the self-healing demo.

Real implementations:
- RESTART: Docker container restart + running/health verification.
- SCALE_UP: vertical Docker CPU/memory update + verification.
- CIRCUIT_BREAK: Docker network isolation when a network is configured.
- REROUTE: executes a configured reroute command after validating a backup.
- ROLLBACK: executes a configured rollback command, optionally requiring approval.

Configuration is read from recovery_config.json beside this file by default.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import docker
import requests


# This file normally lives at:
# src/self_healing_agent/src/UI/backend/recovery_executor.py
BACKEND_DIR = Path(__file__).resolve().parent

# Move from backend -> UI -> src -> self_healing_agent -> shared src,
# then select the sibling Train-Ticket deployment folder.
DEFAULT_TRAIN_TICKET_ROOT = (
    BACKEND_DIR.parents[3] / "train-ticket"
)


# Application service -> database container.
# Update an entry only if your Docker Compose project/container names differ.
DATABASE_CONTAINER_MAP = {
    "ts-auth-service": "train-ticket-ts-auth-mysql-1",
    "ts-order-service": "train-ticket-ts-order-mysql-1",
    "ts-order-other-service": "train-ticket-ts-order-other-mysql-1",
    "ts-travel-service": "train-ticket-ts-travel-mysql-1",
    "ts-payment-service": "train-ticket-ts-payment-mysql-1",
    "ts-station-service": "train-ticket-ts-station-mysql-1",
    #"ts-route-service": "train-ticket-ts-route-mysql-1",
    "ts-train-service": "train-ticket-ts-train-mysql-1",
    "ts-config-service": "train-ticket-ts-config-mysql-1",
    "ts-security-service": "train-ticket-ts-security-mysql-1",
    "ts-contacts-service": "train-ticket-ts-contacts-mysql-1",
    "ts-price-service": "train-ticket-ts-price-mysql-1",
    "ts-inside-payment-service": "train-ticket-ts-inside-payment-mysql-1",
    "ts-notification-service": "train-ticket-ts-notification-mysql-1",
    "ts-consign-service": "train-ticket-ts-consign-mysql-1",
    "ts-food-service": "train-ticket-ts-food-mysql-1",
    "ts-user-service": "train-ticket-ts-user-mysql-1",
    "ts-preserve-service": "train-ticket-ts-preserve-mysql-1",
    "ts-cancel-service": "train-ticket-ts-cancel-mysql-1",
    "ts-rebook-service": "train-ticket-ts-rebook-mysql-1",
}


@dataclass
class RecoveryResult:
    action: str
    service: str
    executed: bool
    verified: bool
    started_at: str
    finished_at: str
    mttr: float | None
    attempt_duration: float
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["recovered"] = self.verified
        return data


class RecoveryExecutor:
    def __init__(self, config_path: str | None = None):
        self.config_path = Path(
            config_path
            or os.getenv("RECOVERY_CONFIG")
            or Path(__file__).with_name("recovery_config.json")
        )
        self.config = self._load_config()
        try:
            self.docker = docker.from_env()
            self.docker.ping()
            self.docker_available = True
            print("RecoveryExecutor connected to Docker.")
        except Exception as exc:
            self.docker = None
            self.docker_available = False
            print(f"Docker unavailable — recovery actions disabled: {exc}")

    def _load_config(self) -> dict[str, Any]:
        defaults = {
            "container_prefix": "train-ticket-",
            "container_suffix": "-1",
            "verification_timeout_seconds": 60,
            "verification_interval_seconds": 3,
            "health_urls": {},
            "scale_up": {"memory": "2g"},
            "circuit_break": {"network": None},
            "adaptive_execution": {
                "enabled": True,
                "prefer_reroute_for_faults": ["service_crash"],
            },
            "reroute": {"services": {}},
            "rollback": {
                "require_confirmation": True,
                "manual_only": True,
            },
        }
        if not self.config_path.exists():
            return defaults
        try:
            loaded = json.loads(self.config_path.read_text(encoding="utf-8"))
            return self._deep_merge(defaults, loaded)
        except Exception as exc:
            print(f"Could not load recovery config: {exc}")
            return defaults

    @staticmethod
    def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        result = dict(base)
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = RecoveryExecutor._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def container_name(self, service: str) -> str:
        return (
            f"{self.config['container_prefix']}"
            f"{service}"
            f"{self.config['container_suffix']}"
        )

    def _backup_is_healthy(
        self,
        service: str,
    ) -> tuple[bool, dict[str, Any]]:
        """Check whether a configured backup container is running and healthy."""
        service_config = (
            self.config
            .get("reroute", {})
            .get("services", {})
            .get(service)
        )

        if not service_config:
            return False, {
                "backup_container": None,
                "backup_available": False,
                "backup_reason": "No reroute configuration for service",
            }

        backup_name = service_config.get("backup_container")

        if not backup_name:
            return False, {
                "backup_container": None,
                "backup_available": False,
                "backup_reason": "No backup container configured",
            }

        if not self.docker_available or self.docker is None:
            return False, {
                "backup_container": backup_name,
                "backup_available": False,
                "backup_reason": "Docker unavailable",
            }

        try:
            backup = self.docker.containers.get(backup_name)
            backup.reload()

            state = backup.attrs.get("State", {})
            status = state.get("Status") or backup.status
            health_status = state.get("Health", {}).get("Status")

            healthy = (
                status == "running"
                and health_status in (None, "healthy")
            )

            return healthy, {
                "backup_container": backup_name,
                "backup_available": healthy,
                "backup_status": status,
                "backup_health_status": health_status or "no_healthcheck",
            }

        except Exception as exc:
            return False, {
                "backup_container": backup_name,
                "backup_available": False,
                "backup_reason": str(exc),
            }

    def choose_final_action(
        self,
        *,
        service: str,
        recommended_action: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Adapt the RL recommendation using the current deployment state.

        Only this case is adapted:
            RL recommends RESTART
            + fault_type == service_crash
            + healthy configured backup exists
            -> execute REROUTE

        ROLLBACK remains ROLLBACK and is handled as a manual action.
        """
        context = context or {}
        fault_type = str(context.get("fault_type", "unknown"))

        adaptive_cfg = self.config.get(
            "adaptive_execution",
            {},
        )
        enabled = bool(adaptive_cfg.get("enabled", True))
        eligible_faults = set(
            adaptive_cfg.get(
                "prefer_reroute_for_faults",
                ["service_crash"],
            )
        )

        decision = {
            "recommended_action": recommended_action,
            "executed_action": recommended_action,
            "action_adapted": False,
            "adaptation_reason": (
                "RL recommendation used without adaptation"
            ),
            "fault_type": fault_type,
        }

        if not enabled:
            decision["adaptation_reason"] = (
                "Adaptive execution is disabled"
            )
            return recommended_action, decision

        if recommended_action != "RESTART":
            decision["adaptation_reason"] = (
                "Only RESTART is eligible for backup-aware adaptation"
            )
            return recommended_action, decision

        if fault_type not in eligible_faults:
            decision["adaptation_reason"] = (
                f"Fault type {fault_type!r} is not eligible for rerouting"
            )
            return recommended_action, decision

        backup_healthy, backup_details = self._backup_is_healthy(
            service
        )
        decision.update(backup_details)

        if not backup_healthy:
            decision["adaptation_reason"] = (
                "No healthy backup is available; RESTART retained"
            )
            return "RESTART", decision

        decision.update({
            "executed_action": "REROUTE",
            "action_adapted": True,
            "adaptation_reason": (
                "Service crash detected and a healthy backup is available"
            ),
        })

        print(
            f"[Adaptive recovery] {service}: "
            f"RESTART -> REROUTE"
        )

        return "REROUTE", decision

    def execute(
        self,
        action: str,
        service: str,
        *,
        approved: bool = False,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = context or {}
        started_ts = time.time()
        started_at = datetime.now().isoformat()

        recommended_action = action

        final_action, action_decision = self.choose_final_action(
            service=service,
            recommended_action=recommended_action,
            context=context,
        )

        handlers = {
            "RESTART": self._restart,
            "SCALE_UP": self._scale_up,
            "REROUTE": self._reroute,
            "ROLLBACK": self._rollback,
        }

        handler = handlers.get(final_action)

        if handler is None:
            result = self._finish(
                final_action,
                service,
                started_ts,
                started_at,
                False,
                False,
                f"Unsupported recovery action: {final_action}",
                {
                    **context,
                    **action_decision,
                },
            )
        else:
            try:
                executed, verified, message, details = handler(
                    service,
                    approved=approved,
                    context=context,
                )
            except Exception as exc:
                executed = False
                verified = False
                message = f"{final_action} failed: {exc}"
                details = {
                    "exception": repr(exc),
                }

            result = self._finish(
                final_action,
                service,
                started_ts,
                started_at,
                executed,
                verified,
                message,
                {
                    **context,
                    **(details or {}),
                    **action_decision,
                },
            )

        result["rl_action"] = recommended_action
        result["recommended_action"] = recommended_action
        result["executed_action"] = final_action
        result["action_adapted"] = bool(
            action_decision.get("action_adapted", False)
        )
        result["adaptation_reason"] = (
            action_decision.get("adaptation_reason")
        )

        manual_required = bool(
            result.get("details", {}).get("manual_required")
            or result.get("details", {}).get("approval_required")
        )
        result["manual_required"] = manual_required

        return result

    def _finish(
        self, action: str, service: str, started_ts: float, started_at: str,
        executed: bool, verified: bool, message: str, details: dict[str, Any]
    ) -> dict[str, Any]:
        finished_at = datetime.now().isoformat()
        attempt_duration = round(time.time() - started_ts, 2)
        result = RecoveryResult(
            action=action,
            service=service,
            executed=executed,
            verified=verified,
            started_at=started_at,
            finished_at=finished_at,
            mttr=attempt_duration if verified else None,
            attempt_duration=attempt_duration,
            message=message,
            details=details,
        )
        return result.to_dict()

    def _docker_guard(self, action: str, service: str):
        if self.docker_available and self.docker is not None:
            return None
        return (
            False,
            False,
            f"{action} not executed because Docker is unavailable",
            {
                "service": service,
                "error": "docker_unavailable",
                "verification_method": "Docker API",
                "verification_status": "Unavailable",
            },
        )

    def _container_running(self, name: str) -> bool:
        try:
            c = self.docker.containers.get(name)
            c.reload()
            return c.status == "running"
        except Exception:
            return False

    def _wait_for_container_running(
        self,
        container_name: str,
        timeout: float | None = None,
        interval: float | None = None,
    ) -> bool:
        """Wait for a container to become running and Docker-healthy, if it has a healthcheck."""
        timeout = float(
            timeout
            if timeout is not None
            else self.config["verification_timeout_seconds"]
        )
        interval = float(
            interval
            if interval is not None
            else self.config["verification_interval_seconds"]
        )
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                container = self.docker.containers.get(container_name)
                container.reload()
                state = container.attrs.get("State", {})
                status = state.get("Status") or container.status
                health_status = state.get("Health", {}).get("Status")

                if status == "running":
                    # Containers without HEALTHCHECK are accepted once running.
                    if health_status in (None, "healthy"):
                        return True
                    if health_status == "unhealthy":
                        return False
            except Exception:
                pass

            time.sleep(interval)

        return False

    def verify_service(self, service: str, *, expected_header_override: dict[str, Any] | None = None,) -> tuple[bool, dict[str, Any]]:
        """Run a service-specific application verification check.

        `expected_header_override` is used by REROUTE to prove that the
        backup instance served the request.
        """
        check = (self.config.get("health_urls", {}).get(service))

        if not check:
            return False, {
                "verification_method": (
                    "No application check configured"
                ),
                "verification_status": "Not configured",
            }

        if isinstance(check, str):
            check = {
                "method": "GET",
                "url": check,
                "expected_status": [200],
            }
        else:
            # Avoid modifying the loaded configuration.
            check = dict(check)

        method = str(
            check.get("method", "GET")
        ).upper()

        url = check.get("url")

        expected_statuses = check.get(
            "expected_status",
            [200],
        )

        if isinstance(expected_statuses, int):
            expected_statuses = [expected_statuses]

        try:
            expected_statuses = [
                int(status)
                for status in expected_statuses
            ]
        except (TypeError, ValueError):
            return False, {
                "verification_method": (
                    f"{method} {url or 'missing-url'}"
                ),
                "verification_status": (
                    "Invalid configuration"
                ),
                "error": (
                    "expected_status must be an integer "
                    "or list of integers"
                ),
            }

        payload = check.get("json")
        params = check.get("params")
        request_headers = check.get(
            "headers",
            {},
        )

        expected_header = (
            expected_header_override
            if expected_header_override is not None
            else check.get("expected_header")
        )

        if not url:
            return False, {
                "verification_method": (
                    "Application HTTP check"
                ),
                "verification_status": (
                    "Invalid configuration"
                ),
                "error": "Missing URL",
            }

        timeout = float(
            self.config[
                "verification_timeout_seconds"
            ]
        )

        interval = float(
            self.config[
                "verification_interval_seconds"
            ]
        )

        deadline = time.time() + timeout

        last_status = None
        last_error = None
        last_response_headers: dict[str, Any] = {}
        attempts = 0

        while time.time() < deadline:
            attempts += 1

            try:
                response = requests.request(
                    method=method,
                    url=url,
                    json=payload,
                    params=params,
                    headers=request_headers,
                    timeout=5,
                )

                last_status = response.status_code
                last_response_headers = dict(
                    response.headers
                )

                status_ok = (
                    response.status_code
                    in expected_statuses
                )

                header_ok = True
                header_name = None
                expected_header_value = None
                actual_header_value = None

                if expected_header:
                    header_name = expected_header.get(
                        "name"
                    )

                    expected_header_value = str(
                        expected_header.get(
                            "value",
                            "",
                        )
                    )

                    if not header_name:
                        return False, {
                            "verification_method": (
                                f"{method} {url}"
                            ),
                            "verification_status": (
                                "Invalid configuration"
                            ),
                            "error": (
                                "expected_header.name "
                                "is missing"
                            ),
                        }

                    actual_header_value = (
                        response.headers.get(
                            header_name
                        )
                    )

                    header_ok = (
                        str(actual_header_value)
                        == expected_header_value
                    )

                if status_ok and header_ok:
                    details: dict[str, Any] = {
                        "verification_method": (
                            f"{method} {url}"
                        ),
                        "verification_status": (
                            f"HTTP "
                            f"{response.status_code}"
                        ),
                        "health_url": url,
                        "attempts": attempts,
                        "http_status": (
                            response.status_code
                        ),
                    }

                    if expected_header:
                        details.update({
                            "expected_header": {
                                "name": header_name,
                                "value": (
                                    expected_header_value
                                ),
                            },
                            "actual_header_value": (
                                actual_header_value
                            ),
                            "routed_backend": (
                                actual_header_value
                            ),
                            "header_verified": True,
                        })

                    return True, details

                if not status_ok:
                    last_error = (
                        f"Expected HTTP "
                        f"{expected_statuses}, "
                        f"received "
                        f"{response.status_code}"
                    )

                elif not header_ok:
                    last_error = (
                        f"Expected header "
                        f"{header_name}="
                        f"{expected_header_value}, "
                        f"received "
                        f"{actual_header_value}"
                    )

            except Exception as exc:
                last_error = str(exc)

            time.sleep(interval)

        failure_details: dict[str, Any] = {
            "verification_method": (
                f"{method} {url}"
            ),
            "verification_status": (
                f"HTTP {last_status}"
                if last_status is not None
                else "Request failed"
            ),
            "health_url": url,
            "attempts": attempts,
            "http_status": last_status,
            "error": last_error,
            "last_response_headers": (
                last_response_headers
            ),
        }

        if expected_header:
            header_name = expected_header.get(
                "name"
            )

            failure_details.update({
                "expected_header": expected_header,
                "actual_header_value": (
                    last_response_headers.get(
                        header_name
                    )
                    if header_name
                    else None
                ),
                "header_verified": False,
            })

        return False, failure_details

    def _wait_until_healthy(
        self,
        service: str,
        container_name: str,
    ) -> tuple[bool, dict[str, Any]]:
        """Verify infrastructure first, then application behaviour."""
        container_ok = self._wait_for_container_running(container_name)

        if not container_ok:
            return False, {
                "container": container_name,
                "infrastructure_verified": False,
                "application_verified": False,
                "verification_method": "Docker container status",
                "verification_status": "Container not running/healthy",
            }

        app_ok, app_details = self.verify_service(service)

        # If no app check is configured, fall back to Docker verification.
        if app_details.get("verification_status") == "Not configured":
            return True, {
                "container": container_name,
                "infrastructure_verified": True,
                "application_verified": None,
                "verification_method": "Docker container status",
                "verification_status": "Running",
            }

        return app_ok, {
            "container": container_name,
            "infrastructure_verified": True,
            "application_verified": app_ok,
            **app_details,
        }


    def restart_database(self, service: str) -> dict[str, Any]:
        """Start/restart the database container mapped to an application service."""
        started_ts = time.time()
        started_at = datetime.now().isoformat()
        database_container = DATABASE_CONTAINER_MAP.get(service)

        if not database_container:
            return self._finish(
                "RESTART_DATABASE",
                service,
                started_ts,
                started_at,
                False,
                False,
                f"No database container mapping configured for {service}",
                {"fault_type": "db_error", "target": None},
            )

        try:
            container = self.docker.containers.get(database_container)
            container.reload()
            previous_status = container.status

            if previous_status in {"exited", "created"}:
                container.start()
                operation = "start"
            elif previous_status in {"dead", "removing"}:
                return self._finish(
                    "RESTART_DATABASE",
                    service,
                    started_ts,
                    started_at,
                    False,
                    False,
                    f"Database container cannot be restarted from status {previous_status}",
                    {
                        "fault_type": "db_error",
                        "container": database_container,
                        "previous_status": previous_status,
                    },
                )
            else:
                container.restart(timeout=20)
                operation = "restart"

            db_running = self._wait_for_container_running(database_container)
            app_verified = False
            verification_details = {
                "verification_method": "Docker container status",
                "verification_status": "Database container running" if db_running else "Database container failed",
                "infrastructure_verified": db_running,
                "application_verified": False,
            }

            if db_running:
                app_verified, app_details = self.verify_service(service)
                if app_details.get("verification_status") == "Not configured":
                    # For DB recovery, application-level verification is preferred.
                    # Without it, we only know the DB container is running.
                    app_verified = False
                verification_details.update(app_details)

            verified = db_running and app_verified

            return self._finish(
                "RESTART_DATABASE",
                service,
                started_ts,
                started_at,
                True,
                verified,
                (
                    f"Database {database_container} recovered and dependent application verified"
                    if verified
                    else f"Database {database_container} was {operation}ed but application verification failed"
                ),
                {
                    "fault_type": "db_error",
                    "container": database_container,
                    "target": database_container,
                    "operation": operation,
                    "previous_status": previous_status,
                    **verification_details,
                },
            )
        except Exception as exc:
            return self._finish(
                "RESTART_DATABASE",
                service,
                started_ts,
                started_at,
                False,
                False,
                f"Database recovery failed: {exc}",
                {
                    "fault_type": "db_error",
                    "container": database_container,
                    "target": database_container,
                    "exception": repr(exc),
                },
            )

    def _restart(self, service: str, **_: Any):
        guard = self._docker_guard("RESTART", service)
        if guard:
            return guard

        name = self.container_name(service)
        container = self.docker.containers.get(name)
        container.restart(timeout=20)

        verified, details = self._wait_until_healthy(service, name)

        return (
            True,
            verified,
            "Container restarted and application verification passed"
            if verified
            else "Container restarted but verification failed",
            details,
        )

    def _scale_up(self, service: str, **_: Any):
        name = self.container_name(service)
        settings = self.config["scale_up"]
        memory = str(settings.get("memory", "2g"))
        command = [
            "docker", "update", f"--memory={memory}", f"--memory-swap={memory}", name
            
        ]
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0:
            return False, False, proc.stderr.strip() or "docker update failed", {
                "command": command,
            }
        verified, details = self._wait_until_healthy(service, name)
        details.update({"memory": memory})
        return (
            True,
            verified,
            "Resources increased and application verification passed"
            if verified
            else "Resources increased but verification failed",
            details,
        )

    def _reroute(self,service: str,**_: Any,):
        """Switch public traffic to a configured backup through Nginx."""

        guard = self._docker_guard("REROUTE",service,)

        if guard:
            return guard

        reroute_config = self.config.get(
            "reroute",
            {},
        )

        service_config = (
            reroute_config
            .get("services", {})
            .get(service)
        )

        if not service_config:
            return (
                False,
                False,
                f"REROUTE is not configured for {service}",
                {
                    "service": service,
                    "verification_method": (
                        "Reroute configuration"
                    ),
                    "verification_status": (
                        "Not configured"
                    ),
                },
            )

        backup_name = service_config.get(
            "backup_container"
        )

        command = service_config.get(
            "switch_command"
        )

        expected_header = service_config.get(
            "expected_header",
            {
                "name": "X-Backend",
                "value": "backup",
            },
        )

        missing = []

        if not backup_name:
            missing.append("backup_container")

        if not command:
            missing.append("switch_command")

        if missing:
            return (
                False,
                False,
                "REROUTE configuration is incomplete",
                {
                    "service": service,
                    "missing": missing,
                    "verification_method": (
                        "Reroute configuration"
                    ),
                    "verification_status": (
                        "Invalid configuration"
                    ),
                },
            )

        run_command = (
            command
            if isinstance(command, list)
            else shlex.split(command)
        )

        try:
            # Step 1: Ensure the backup instance is running.
            backup = self.docker.containers.get(
                backup_name
            )

            backup.reload()
            previous_backup_status = backup.status

            if backup.status != "running":
                backup.start()

            backup_running = (
                self._wait_for_container_running(
                    backup_name
                )
            )

            if not backup_running:
                return (
                    True,
                    False,
                    "Backup container did not become ready",
                    {
                        "service": service,
                        "backup": backup_name,
                        "previous_backup_status": (
                            previous_backup_status
                        ),
                        "verification_method": (
                            "Backup container status"
                        ),
                        "verification_status": "Failed",
                    },
                )

            # Step 2: Switch the Nginx upstream.
            process = subprocess.run(
                run_command,
                capture_output=True,
                text=True,
                check=False,
            )

            if process.returncode != 0:
                return (
                    False,
                    False,
                    (
                        process.stderr.strip()
                        or process.stdout.strip()
                        or "Nginx route switch failed"
                    ),
                    {
                        "service": service,
                        "backup": backup_name,
                        "command": run_command,
                        "stdout": (
                            process.stdout.strip()
                        ),
                        "stderr": (
                            process.stderr.strip()
                        ),
                        "return_code": (
                            process.returncode
                        ),
                        "verification_method": (
                            "Nginx route switch"
                        ),
                        "verification_status": "Failed",
                    },
                )

            # Step 3: Use the same public URL, but require the
            # response header to identify the backup.
            verified, verification_details = (
                self.verify_service(
                    service,
                    expected_header_override=(
                        expected_header
                    ),
                )
            )

            details = {
                "service": service,
                "backup": backup_name,
                "target": backup_name,
                "route_target": "backup",
                "previous_backup_status": (
                    previous_backup_status
                ),
                "command": run_command,
                "stdout": process.stdout.strip(),
                "stderr": process.stderr.strip(),
                **verification_details,
            }

            return (
                True,
                verified,
                (
                    "Healthy backup is available. Traffic rerouted to backup and "
                    "application verification passed"
                    if verified
                    else
                    "Traffic switch completed, but "
                    "backup verification failed"
                ),
                details,
            )

        except Exception as exc:
            return (
                False,
                False,
                f"REROUTE failed: {exc}",
                {
                    "service": service,
                    "backup": backup_name,
                    "command": run_command,
                    "exception": repr(exc),
                    "verification_method": (
                        "Reroute execution"
                    ),
                    "verification_status": "Failed",
                },
            )

    def _rollback(
        self,
        service: str,
        *,
        approved: bool = False,
        **_: Any,
    ):
        """Escalate rollback to an operator instead of executing it automatically.

        Rollback is retained in the RL action space because it is appropriate for
        deterministic logic, configuration, and deployment regressions. However,
        automatic execution is intentionally disabled because it may revert valid
        changes or introduce version/schema incompatibilities.
        """
        cfg = self.config.get("rollback", {})
        require_confirmation = bool(
            cfg.get("require_confirmation", True)
        )
        manual_only = bool(
            cfg.get("manual_only", True)
        )

        if manual_only or (require_confirmation and not approved):
            return (
                False,
                False,
                "Manual rollback required",
                {
                    "service": service,
                    "manual_required": True,
                    "approval_required": True,
                    "recovery_status": "manual_required",
                    "verification_method": "Operator approval",
                    "verification_status": "Pending manual rollback",
                    "risk_reason": (
                        "Automatic rollback is disabled because it may "
                        "revert valid changes or create compatibility issues."
                    ),
                },
            )

        return (
            False,
            False,
            "Approved rollback execution is not implemented",
            {
                "service": service,
                "manual_required": True,
                "approval_required": False,
                "recovery_status": "manual_required",
                "verification_method": "Rollback execution",
                "verification_status": "Not implemented",
            },
        )