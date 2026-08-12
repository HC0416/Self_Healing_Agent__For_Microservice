from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any


NGINX_CONTAINER = "train-ticket-nginx-1"

TRAIN_TICKET_ROOT = (
    Path(__file__).resolve().parents[4]
    / "train-ticket"
)

NGINX_CONFIG_DIR = (
    TRAIN_TICKET_ROOT
    / "nginx"
)


ROUTES: dict[str, dict[str, Any]] = {
    "ts-auth-service": {
        "listen_port": 12340,
        "upstream_name": "auth_backend",
        "config_name": "auth.conf",

        "primary": (
            "train-ticket-ts-auth-service-1:12340"
        ),

        "backup": (
            "train-ticket-ts-auth-service-backup-1:12340"
        ),
    },
}


def run_command(
    command: list[str],
) -> subprocess.CompletedProcess[str]:
    """Run a command and capture its output."""

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )


def build_nginx_config(
    *,
    upstream_name: str,
    listen_port: int,
    backend: str,
    route_target: str,
) -> str:
    """Build the Nginx configuration for one route target."""

    return f"""upstream {upstream_name} {{
    server {backend};
}}

server {{
    listen {listen_port};

    location / {{
        proxy_pass http://{upstream_name};

        proxy_connect_timeout 3s;
        proxy_read_timeout 20s;
        proxy_send_timeout 20s;

        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        add_header X-Backend {route_target} always;
    }}
}}
"""


def ensure_nginx_running() -> None:
    """Confirm that the Nginx container is running."""

    result = run_command([
        "docker",
        "inspect",
        "-f",
        "{{.State.Running}}",
        NGINX_CONTAINER,
    ])

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or result.stdout.strip()
            or (
                f"Could not inspect Nginx container "
                f"{NGINX_CONTAINER}"
            )
        )

    if result.stdout.strip().lower() != "true":
        raise RuntimeError(
            f"Nginx container is not running: "
            f"{NGINX_CONTAINER}"
        )


def test_nginx_configuration() -> None:
    """Run nginx -t inside the container."""

    result = run_command([
        "docker",
        "exec",
        NGINX_CONTAINER,
        "nginx",
        "-t",
    ])

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or result.stdout.strip()
            or "Nginx configuration test failed"
        )


def reload_nginx() -> None:
    """Reload Nginx without restarting the container."""

    result = run_command([
        "docker",
        "exec",
        NGINX_CONTAINER,
        "nginx",
        "-s",
        "reload",
    ])

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or result.stdout.strip()
            or "Nginx reload failed"
        )


def switch_route(
    service: str,
    target: str,
) -> dict[str, Any]:
    """Switch a service between its primary and backup backend."""

    route = ROUTES.get(service)

    if route is None:
        raise ValueError(
            f"Reroute is not configured for {service}"
        )

    if target not in {"primary", "backup"}:
        raise ValueError(
            "target must be either 'primary' or 'backup'"
        )

    ensure_nginx_running()

    backend = route[target]

    config_text = build_nginx_config(
        upstream_name=route["upstream_name"],
        listen_port=route["listen_port"],
        backend=backend,
        route_target=target,
    )

    NGINX_CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    active_config = (
        NGINX_CONFIG_DIR
        / route["config_name"]
    )

    backup_config = active_config.with_suffix(
        active_config.suffix + ".before-switch"
    )

    # Keep the current working configuration so it can be
    # restored if nginx -t or reload fails.
    if active_config.exists():
        shutil.copy2(
            active_config,
            backup_config,
        )

    try:
        # Write directly to the host-mounted Nginx config.
        active_config.write_text(
            config_text,
            encoding="utf-8",
        )

        # The host nginx directory must be mounted as:
        # ./nginx:/etc/nginx/conf.d
        test_nginx_configuration()
        reload_nginx()

    except Exception:
        # Restore the previous working configuration.
        if backup_config.exists():
            shutil.copy2(
                backup_config,
                active_config,
            )

            # Try to restore the previous active Nginx state.
            try:
                test_nginx_configuration()
                reload_nginx()
            except Exception:
                pass

        raise

    return {
        "service": service,
        "target": target,
        "backend": backend,
        "nginx_container": NGINX_CONTAINER,
        "config_file": str(active_config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Switch Nginx traffic between a primary "
            "and backup microservice instance."
        )
    )

    parser.add_argument(
        "service",
        help="Service whose route should be changed",
    )

    parser.add_argument(
        "target",
        choices=[
            "primary",
            "backup",
        ],
        help="Backend target",
    )

    args = parser.parse_args()

    result = switch_route(
        service=args.service,
        target=args.target,
    )

    print(
        f"Traffic for {result['service']} switched "
        f"to {result['target']} "
        f"({result['backend']})"
    )

    print(
        f"Nginx config updated: "
        f"{result['config_file']}"
    )


if __name__ == "__main__":
    main()