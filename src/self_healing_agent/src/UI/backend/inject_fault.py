import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

""" 
sh -c "tc qdisc del dev eth0 root"
python UI\backend\inject_fault.py reroute or scale-up,restart or restore or reroute
"""


PRIMARY_CONTAINER = "train-ticket-ts-auth-service-1"
BACKUP_CONTAINER = "train-ticket-ts-auth-service-backup-1"

LOGIN_URL = "http://localhost:12340/api/v1/users/login"

LOGIN_DATA = {
    "username": "admin",
    "password": "222222",
}

NETWORK_DELAY_MS = 2000

def run_command(command: list[str]) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or result.stdout.strip()
            or f"Command failed: {' '.join(command)}"
        )

    output = result.stdout.strip()

    if output:
        print(output)

    return output


def container_status(container: str) -> str:
    return run_command([
        "docker",
        "inspect",
        "--format",
        "{{.State.Status}}",
        container,
    ])


def route_to_primary() -> None:
    print("Switching Nginx traffic to primary...")

    run_command([
        "python",
        "-m",
        "UI.backend.switch_route",
        "ts-auth-service",
        "primary",
    ])


# ============================================================
# SCALE-UP FAULT
# ============================================================

def limit_cpu(cpus: float = 0.25) -> None:
    print(
        f"Limiting {PRIMARY_CONTAINER} "
        f"to {cpus} CPU..."
    )

    cpu_period = 100000
    cpu_quota = int(cpu_period * cpus)

    run_command([
        "docker",
        "update",
        f"--cpu-period={cpu_period}",
        f"--cpu-quota={cpu_quota}",
        PRIMARY_CONTAINER,
    ])

    print(
        "CPU fault injected: "
        f"period={cpu_period}, "
        f"quota={cpu_quota}"
    )


def restore_cpu(cpus: float = 2.0) -> None:
    print(
        f"Restoring {PRIMARY_CONTAINER} "
        f"to {cpus} CPUs..."
    )

    cpu_period = 100000
    cpu_quota = int(cpu_period * cpus)

    run_command([
        "docker",
        "update",
        f"--cpu-period={cpu_period}",
        f"--cpu-quota={cpu_quota}",
        PRIMARY_CONTAINER,
    ])

    print("CPU restored.")


def fresh_login(
    request_number: int,
) -> dict[str, Any]:
    started = time.time()

    try:
        response = requests.post(
            LOGIN_URL,
            json=LOGIN_DATA,
            timeout=30,
        )

        elapsed = round(
            time.time() - started,
            2,
        )

        try:
            message = response.json().get("msg")
        except Exception:
            message = response.text[:100]

        result = {
            "request": request_number,
            "ok": response.status_code == 200,
            "status": response.status_code,
            "message": message,
            "backend": response.headers.get(
                "X-Backend"
            ),
            "elapsed": elapsed,
        }

        print(
            f"[{request_number:02d}] "
            f"HTTP={result['status']} "
            f"backend={result['backend']} "
            f"time={result['elapsed']:.2f}s "
            f"msg={result['message']}"
        )

        return result

    except Exception as exc:
        elapsed = round(
            time.time() - started,
            2,
        )

        result = {
            "request": request_number,
            "ok": False,
            "status": None,
            "message": str(exc),
            "backend": None,
            "elapsed": elapsed,
        }

        print(
            f"[{request_number:02d}] "
            f"FAILED after {elapsed:.2f}s: "
            f"{exc}"
        )

        return result


def generate_login_load(
    count: int = 20,
    workers: int = 8,
) -> list[dict[str, Any]]:
    print(
        f"\nSending {count} login requests "
        f"with {workers} workers..."
    )

    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(
        max_workers=workers
    ) as pool:
        futures = [
            pool.submit(
                fresh_login,
                request_number,
            )
            for request_number in range(
                1,
                count + 1,
            )
        ]

        for future in as_completed(futures):
            results.append(
                future.result()
            )

    successes = sum(
        1
        for result in results
        if result["ok"]
    )

    elapsed_values = [
        result["elapsed"]
        for result in results
    ]

    average = (
        sum(elapsed_values) / len(elapsed_values)
        if elapsed_values
        else 0
    )

    print("\nLoad summary")
    print(f"Success: {successes}/{count}")
    print(
        "Minimum time: "
        f"{min(elapsed_values, default=0):.2f}s"
    )
    print(
        "Average time: "
        f"{average:.2f}s"
    )
    print(
        "Maximum time: "
        f"{max(elapsed_values, default=0):.2f}s"
    )

    return results


def inject_scale_up_fault() -> None:
    print("=" * 60)
    print("SCALE-UP FAULT INJECTION")
    print("=" * 60)

    route_to_primary()

    # print("\nPreflight login:")
    # preflight = fresh_login(0)

    # if not preflight["ok"]:
    #     raise RuntimeError(
    #         "Preflight login failed. "
    #         "Do not inject CPU pressure."
    #     )

    limit_cpu(0.25)

    time.sleep(2)

    generate_login_load(
        count=20,
        workers=8,
    )

def inject_restart_fault() -> None:
    print("=" * 60)
    print("RESTART FAULT INJECTION")
    print("=" * 60)

    route_to_primary()

    run_command([
        "docker",
        "stop",
        BACKUP_CONTAINER,
    ])

    print(
        f"Backup status: "
        f"{container_status(BACKUP_CONTAINER)}"
    )

    run_command([
        "docker",
        "stop",
        PRIMARY_CONTAINER,
    ])

    print(
        f"Primary status: "
        f"{container_status(PRIMARY_CONTAINER)}"
    )

    print(
        "\nCrash injected. The real monitor should detect "
        "the stopped primary through Docker."
    )

# ============================================================
# REROUTE FAULT
# ============================================================
def route_to_backup() -> None:
    print("Switching Nginx traffic to backup...")

    run_command([
        "python",
        "-m",
        "UI.backend.switch_route",
        "ts-auth-service",
        "backup",
    ])


def wait_for_backend(expected_backend: str,timeout: int = 90,) -> bool:
    print(f"Waiting for traffic to reach "f"{expected_backend}...")

    deadline = time.time() + timeout

    while time.time() < deadline:
        result = fresh_login(0)

        if (result.get("ok")and result.get("backend")== expected_backend):
            print(f"Verified X-Backend="f"{expected_backend}")
            return True

        time.sleep(3)

    return False

def login_once():
    resp = requests.post("http://localhost:12340/api/v1/users/login",
                    json={"username":"admin","password":"222222"}, timeout=30)
    print(resp.status_code, resp.json().get("msg"))

# ============================================================
# REROUTE FAULT — NETWORK DELAY
# ============================================================

def install_tc_if_missing() -> None:

    check = subprocess.run(
        [
            "docker",
            "exec",
            PRIMARY_CONTAINER,
            "sh",
            "-c",
            "command -v tc",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if check.returncode == 0:
        return

    print("Installing iproute2 inside primary container...")

    run_command([
        "docker",
        "exec",
        PRIMARY_CONTAINER,
        "sh",
        "-c",
        (
            "apt-get update && "
            "apt-get install -y iproute2"
        ),
    ])


def inject_network_delay(delay_ms: int = NETWORK_DELAY_MS,) -> None:

    install_tc_if_missing()

    print(f"Injecting {delay_ms} ms network delay into {PRIMARY_CONTAINER}...")

    run_command([
        "docker",
        "exec",
        PRIMARY_CONTAINER,
        "sh",
        "-c",
        (
            "tc qdisc replace dev eth0 root "
            f"netem delay {delay_ms}ms"
        ),
    ])

    print("Network delay injected.")


def clear_network_delay() -> None:

    print("Removing network delay...")

    result = subprocess.run(
        [
            "docker",
            "exec",
            PRIMARY_CONTAINER,
            "sh",
            "-c",
            (
                "tc qdisc del dev eth0 root "
                "2>/dev/null || true"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(
            "Warning: network delay could not be removed:",
            result.stderr.strip(),
        )
    else:
        print("Network delay removed.")


def network_delay_status() -> str:
    """Show the active tc configuration."""

    return run_command([
        "docker",
        "exec",
        PRIMARY_CONTAINER,
        "sh",
        "-c",
        "tc qdisc show dev eth0",
    ])


def inject_reroute_fault() -> None:
    """Inject network delay and generate traces for REROUTE.
    """

    print("=" * 60)
    print("REROUTE FAULT INJECTION")
    print("=" * 60)

    # Both services must be running.
    for container in [
        PRIMARY_CONTAINER,
        BACKUP_CONTAINER,
    ]:
        status = container_status(container)

        if status != "running":
            print(f"Starting {container}...")

            run_command([
                "docker",
                "start",
                container,
            ])

    print("\nWaiting for containers to initialise...")
    time.sleep(20)

    # Begin the experiment on the primary.
    route_to_primary()

    inject_network_delay(delay_ms=NETWORK_DELAY_MS)

    print("\nCurrent network rule:")
    network_delay_status()

    time.sleep(2)

    generate_login_load(count=8,workers=4,)

# ============================================================
# RESTORE ENVIRONMENT
# ============================================================

def restore_environment() -> None:
    print("=" * 60)
    print("RESTORING DEMO ENVIRONMENT")
    print("=" * 60)

    for container in [
        PRIMARY_CONTAINER,
        BACKUP_CONTAINER,
    ]:
        status = container_status(container)

        if status != "running":
            print(f"Starting {container}...")

            run_command([
                "docker",
                "start",
                container,
            ])

    restore_cpu(2.0)

    print(
        "\nWaiting for containers to initialise..."
    )
    time.sleep(20)

    route_to_primary()

    print("\nVerification login:")
    result = fresh_login(0)

    if not result["ok"]:
        print(
            "Warning: login verification failed. "
            "The service may still be starting."
        )

    print("Environment restoration completed.")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=("Inject Train-Ticket faults for the real self-healing monitor."))

    parser.add_argument("fault",choices=["scale-up","restart","restore", "reroute"])

    args = parser.parse_args()

    if args.fault == "scale-up":
        inject_scale_up_fault()

    elif args.fault == "restart":
        inject_restart_fault()

    elif args.fault == "reroute":
        inject_reroute_fault()

    # else:
    #     restore_environment()


if __name__ == "__main__":
    main()