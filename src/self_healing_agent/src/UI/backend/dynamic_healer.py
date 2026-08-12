"""
Dynamic self-healing agent.
Continuously monitors ALL services in SkyWalking,
detects anomalies with GNN, and executes A3 recovery actions.

Usage:
    python dynamic_healer.py           # monitor all services
    python dynamic_healer.py --once    # single scan pass
    python dynamic_healer.py --service ts-auth-service  # monitor one service
"""

import sys, os, time, argparse, subprocess, json
import torch
import torch.nn as nn
import requests
import numpy as np
import docker
from datetime import datetime, timedelta
from collections import defaultdict
from common.text_strings import Path_Strings, Url_Strings
from UI.backend.recovery_executor import RecoveryExecutor
from common.llm import LLM
from common.text_strings import Model_Strings
from models.model.deeptralog_model import DeepTraLogModel
from models.utils.dataset_with_logs import GloVeEmbedder
from torch_geometric.data import Data


# ──────────────────────────────────────────────────────────
# 1. CONFIG
# ──────────────────────────────────────────────────────────
MODELS_PATH  = Path_Strings.MODELS_PATH
GNN_PATH     = os.path.join(MODELS_PATH, 'results', 'M4_ggnn_normal_only.pt')
GLOVE_PATH   = Path_Strings.GLOVE_PATH
A3_PATH      = Path_Strings.A3_PATH

SW_GRAPHQL   = Url_Strings.SW_GRAPHQL
SCAN_INTERVAL = 30   # seconds between each scan
NOISE_LEVEL   = 0.0  # keep live demo deterministic; no random feature noise  

# Persist calibrated per-service baselines beside this file.
BASELINE_STORE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'service_baselines.json'
)

# Services to skip (databases, infrastructure)
SKIP_SERVICES = {
    "ts-auth-mysql", "ts-order-mysql", "ts-travel-mysql",
    "ts-payment-mysql", "ts-station-mysql", #"ts-route-mysql",
    "ts-train-mysql", "ts-config-mysql", "ts-security-mysql",
    "ts-contacts-mysql", "ts-price-mysql", "ts-inside-payment-mysql",
    "ts-notification-mysql", "ts-consign-mysql", "ts-food-mysql",
}

# ──────────────────────────────────────────────────────────
# 2. LOAD MODELS
# ──────────────────────────────────────────────────────────
sys.path.insert(0, MODELS_PATH)

def load_models():
    print("Loading GNN model...")
    gnn_ckpt = torch.load(GNN_PATH, map_location="cpu", weights_only=False,)

    gnn_model = DeepTraLogModel(
        in_channels=gnn_ckpt["in_channels"],
        hidden=gnn_ckpt["hidden"],
        latent=gnn_ckpt["latent"],
        num_layers=gnn_ckpt["num_layers"],
        model_type=gnn_ckpt["model_type"],
    )

    gnn_model.load_state_dict(gnn_ckpt["model_state"])
    gnn_model.svdd.centre = gnn_ckpt["centre"]
    gnn_model.eval()
    print("GNN loaded")

    print("Loading GloVe...")
    embedder = GloVeEmbedder(GLOVE_PATH)
    print("GloVe loaded")

    print("Loading A3 policy...")

    a3_ckpt = torch.load(A3_PATH,map_location="cpu",weights_only=False,)

    expected_actions = [
        "RESTART",
        "SCALE_UP",
        "REROUTE",
        "ROLLBACK",
    ]

    actions = a3_ckpt.get("actions", expected_actions)

    if actions != expected_actions:
        raise ValueError(
            "Incompatible A3 checkpoint. "
            f"Expected actions {expected_actions}, "
            f"but checkpoint contains {actions}. "
            "Run the updated four-action training notebook and replace A3_PATH "
            "with the newly generated checkpoint."
        )

    state_dim = int(a3_ckpt.get("state_dim", 10))
    action_dim = int(a3_ckpt.get("action_dim", len(actions)))

    if action_dim != len(actions):
        raise ValueError(
            "Invalid A3 checkpoint metadata: "
            f"action_dim={action_dim}, but {len(actions)} actions were saved."
        )

    class PolicyNetA3(nn.Module):
        def __init__(self, state_dim=10, action_dim=4):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
            )

        def forward(self, x):
            return torch.softmax(self.net(x),dim=-1)

        def logits(self, x):
            return self.net(x)

    a3_policy = PolicyNetA3(state_dim=state_dim,action_dim=action_dim)

    try:
        a3_policy.load_state_dict(a3_ckpt["model_state"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load the A3 policy checkpoint. "
            "The checkpoint may still contain the old five-action output layer. "
            "Retrain A3 with the four-action notebook and update A3_PATH."
        ) from exc

    a3_policy.eval()

    print(f"A3 loaded   "f"State dim: {state_dim}   "f"Action dim: {action_dim}   "f"Actions: {actions}")

    llm, tokenizer = LLM.load_reasoning_llm(Model_Strings.llm_3_model)

    return (gnn_model, embedder, a3_policy, actions, llm, tokenizer)

# ──────────────────────────────────────────────────────────
# 3. FEATURE NAMES + EXPLAINER
# ──────────────────────────────────────────────────────────
FEATURE_NAMES = [
    'anomaly_score',    # [0] GNN SVDD distance
    'service_crash',    # [1] service-level failure signal
    'slow_response',    # [2] Spans > 3000ms
    'overload_timeout', # [3] timeout/load/overload signal from trace symptoms
    'network_delay',    # [4] HTTP slow, DB normal
    'is_anomaly',       # [5] always 1
    'db_error',         # [6] MySQL/JDBC/HikariCP dependency error
    'logic_error',      # [7] Errors not from DB/HikariCP
    'async_error',      # [8] Sequence/async issues
    'error_ratio',      # [9] Error spans / total spans
]


def infer_fault_type_from_state(state):
    """Return the UI fault label from the 10-dim RL state.

    DB/HikariCP signals are prioritised so a dependency failure is labelled
    db_error even when it also causes a service-level failure signal.
    """
    feats = dict(zip(FEATURE_NAMES, state))
    priority = [
        'db_error',
        'service_crash',
        'network_delay',
        'overload_timeout',
        'slow_response',
        'logic_error',
        'async_error',
    ]
    for feat in priority:
        if float(feats.get(feat, 0.0)) > 0.05:
            return feat
    return 'anomaly'



# ──────────────────────────────────────────────────────────
# 4. SERVICE MAP
# ──────────────────────────────────────────────────────────
SW_SERVICE_MAP = {
    "ts-auth-service": 0,            "ts-order-service": 1,
    "ts-order-other-service": 2,     "ts-station-service": 3,
    "ts-travel-service": 4,          "ts-execute-service": 5,
    "ts-preserve-service": 6,        "ts-basic-service": 7,
    "ts-cancel-service": 8,          "ts-rebook-service": 9,
    "ts-inside-payment-service": 10, "ts-payment-service": 11,
    "ts-route-service": 12,          "ts-route-plan-service": 13,
    "ts-train-service": 14,          "ts-seat-service": 15,
    "ts-user-service": 16,           "ts-config-service": 17,
    "ts-security-service": 18,       "ts-notification-service": 19,
    "ts-food-service": 20,           "ts-contacts-service": 21,
    "ts-consign-service": 22,        "ts-price-service": 23,
    "ts-news-service": 24,           "ts-verification-code-service":25,
}

# ──────────────────────────────────────────────────────────
# 4. SKYWALKING HELPERS
# ──────────────────────────────────────────────────────────
def sw_duration(hours_back=2):
    now   = datetime.now()
    start = now - timedelta(hours=hours_back)
    return {"start": start.strftime("%Y-%m-%d %H"),
            "end":   now.strftime("%Y-%m-%d %H")}

def get_all_services():
    dur = sw_duration(24)
    query = f"""
    {{ getAllServices(duration: {{
        start: "{dur['start']}" end: "{dur['end']}" step: HOUR
    }}) {{ id name }} }}
    """
    try:
        resp = requests.post(SW_GRAPHQL, json={"query": query},
                            headers={"Content-Type": "application/json"},
                            timeout=10)
        return {s["name"]: s["id"]
                for s in resp.json()["data"]["getAllServices"]
                if s["name"] not in SKIP_SERVICES}
    except:
        return {}

def get_latest_trace(service_id):
    dur = sw_duration(2)
    query = f"""
    {{
      queryBasicTraces(condition: {{
        queryDuration: {{ start: "{dur['start']}" end: "{dur['end']}" step: HOUR }}
        traceState: ALL
        queryOrder: BY_START_TIME
        serviceId: "{service_id}"
        paging: {{ pageNum: 1, pageSize: 5 }}
      }}) {{ traces {{ traceIds duration start }} }}
    }}
    """
    try:
        resp   = requests.post(SW_GRAPHQL, json={"query": query},
                              headers={"Content-Type": "application/json"},
                              timeout=10)
        traces = resp.json()["data"]["queryBasicTraces"]["traces"]
        if not traces:
            return None
        return max(traces, key=lambda t: int(t["start"]))
    except:
        return None

def get_spans(trace_id):
    query = f"""
    {{
      queryTrace(traceId: "{trace_id}") {{
        spans {{
          traceId segmentId spanId parentSpanId
          serviceCode startTime endTime endpointName isError type
        }}
      }}
    }}
    """
    try:
        resp = requests.post(SW_GRAPHQL, json={"query": query},
                            headers={"Content-Type": "application/json"},
                            timeout=10)
        return resp.json()["data"]["queryTrace"]["spans"]
    except:
        return []

# ──────────────────────────────────────────────────────────
# 5. GNN SCORING
# ──────────────────────────────────────────────────────────
def score_spans(spans, gnn_model, embedder):
    if not spans:
        return None, None

    span_id_to_idx = {s['spanId']: i for i, s in enumerate(spans)}
    durations = [s['endTime'] - s['startTime'] for s in spans]
    mean_dur  = max(np.mean(durations), 1.0)
    err_ratio = sum(
        1 for s in spans
        if s['isError'] or (s['endTime'] - s['startTime']) > 3000
    ) / max(len(spans), 1)

    children_count = defaultdict(int)
    for s in spans:
        pid = s['parentSpanId']
        if pid >= 0 and pid in span_id_to_idx:
            children_count[span_id_to_idx[pid]] += 1

    type_map = {"Entry": 0, "Exit": 1, "Local": 2}
    nodes, node_meta, edges_src, edges_dst = [], [], [], []

    for i, span in enumerate(spans):
        svc      = span.get('serviceCode', 'unknown')
        duration = span['endTime'] - span['startTime']
        is_error = 1.0 if span['isError'] or duration > 3000 else 0.0
        svc_id   = SW_SERVICE_MAP.get(svc, 0)
        type_id  = type_map.get(span.get('type', 'Exit'), 1)
        glove    = embedder.embed(f"service {svc_id} type {type_id}")

        feat       = np.zeros(306, dtype=np.float32)
        feat[:300] = glove
        feat[300]  = is_error
        feat[301]  = 0.0
        feat[302]  = i
        feat[303]  = float(children_count.get(i, 0))
        feat[304]  = duration / mean_dur
        feat[305]  = err_ratio

        nodes.append(feat)
        node_meta.append({
            "service":  svc,
            "endpoint": span.get('endpointName', ''),
            "duration": duration,
            "is_error": span['isError'],
        })
        pid = span['parentSpanId']
        if pid >= 0 and pid in span_id_to_idx:
            edges_src.append(span_id_to_idx[pid])
            edges_dst.append(i)

    x = torch.tensor(np.stack(nodes), dtype=torch.float)
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long) \
                 if edges_src else torch.zeros((2, 0), dtype=torch.long)
    data           = Data(x=x, edge_index=edge_index)
    data.batch     = torch.zeros(len(nodes), dtype=torch.long)
    data.node_meta = node_meta

    with torch.no_grad():
        scores, _, node_scores = gnn_model(data)
    score    = scores.item()
    root_idx = node_scores.argmax().item()
    return score, node_meta[root_idx]['endpoint']

# ──────────────────────────────────────────────────────────
# 6. STATE BUILDER
# ──────────────────────────────────────────────────────────
def container_has_netem_delay(service_name):
    """Return True when the service container has an active tc netem delay."""
    try:
        container_name = recovery_executor.container_name(service_name)
        container = recovery_executor.docker.containers.get(container_name)
        container.reload()
        if container.status != "running":
            return False
        result = container.exec_run(
            ["sh", "-c", "tc qdisc show dev eth0 2>/dev/null || true"],
            stdout=True,
            stderr=True,
        )
        output = result.output.decode("utf-8", errors="ignore").lower()
        return "netem" in output and "delay" in output
    except Exception:
        return False


def build_state_from_trace(score, spans, service_name):
    """Build the 10-dimensional RL state from trace and Docker symptoms."""
    durations = [int(s["endTime"] - s["startTime"]) for s in spans]
    max_dur = max(durations) if durations else 0
    span_count = max(len(spans), 1)
    err_count = sum(1 for s in spans if bool(s.get("isError")))
    error_ratio = min(err_count / span_count, 1.0)

    endpoints = [str(s.get("endpointName", "") or "") for s in spans]
    endpoint_text = " ".join(endpoints).lower()

    def span_duration(span):
        return int(span["endTime"] - span["startTime"])

    def is_db_endpoint(endpoint):
        value = endpoint.lower()
        return any(k in value for k in (
            "mysql", "jdbc", "hikari", "preparedstatement",
            "statement/", "connection/getconnection", "connection/close",
        ))

    def is_http_endpoint(endpoint):
        value = endpoint.lower()
        return any(k in value for k in (
            "http", "/api/", "users/login", "user/login", "login",
            "post:", "get:", "post ", "get ",
            "auth.controller", "authcontroller",
        ))

    timeout_keywords = (
        "timeout", "timed out", "read timed out", "connect timed out",
        "connection reset", "broken pipe", "connection refused", "unavailable",
    )
    overload_keywords = (
        "overload", "overloaded", "too many requests", "too many",
        "thread pool", "queue full", "rejected execution",
        "resource exhausted", "denial",
    )
    config_keywords = (
        "config", "configuration", "classpath", "jar", "max-content-length",
        "null", "wrong", "missing", "invalid", "format", "schema",
        "column", "locale",
    )
    async_keywords = (
        "async", "sequence", "order", "ordered", "race", "concurrent",
        "callback", "future", "thread",
    )

    slow_count = sum(1 for d in durations if d > 3000)
    very_slow_count = sum(1 for d in durations if d > 10000)
    slow_response = max_dur > 3000

    has_timeout_keyword = any(k in endpoint_text for k in timeout_keywords)
    has_overload_keyword = any(k in endpoint_text for k in overload_keywords)
    has_config_keyword = any(k in endpoint_text for k in config_keywords)
    has_async_keyword = any(k in endpoint_text for k in async_keywords)

    http_slow = any(
        is_http_endpoint(ep) and span_duration(s) > 3000
        for ep, s in zip(endpoints, spans)
    )
    db_error_span = any(
        is_db_endpoint(ep) and bool(s.get("isError"))
        for ep, s in zip(endpoints, spans)
    )
    db_timeout_span = any(
        is_db_endpoint(ep)
        and span_duration(s) > 3000
        and any(k in ep.lower() for k in timeout_keywords)
        for ep, s in zip(endpoints, spans)
    )

    db_error = db_error_span or db_timeout_span
    netem_delay_active = container_has_netem_delay(service_name)
    network_delay = (
        netem_delay_active
        and slow_response
        and not db_error
        and err_count == 0
    )
    service_crash = err_count > 0 and not db_error and not network_delay
    overload_timeout = (
        not network_delay
        and not db_error
        and not service_crash
        and (
            has_overload_keyword
            or very_slow_count > 0
            or (slow_count >= 2 and span_count >= 4)
            or (max_dur > 2500 and slow_response)
            or (has_timeout_keyword and not http_slow)
        )
    )
    logic_config_error = has_config_keyword and not db_error and not network_delay
    async_order_error = has_async_keyword and not db_error and not network_delay

    state = [
        min(float(score), 1.0),
        1.0 if service_crash else 0.0,
        1.0 if slow_response else 0.0,
        1.0 if overload_timeout else 0.0,
        1.0 if network_delay else 0.0,
        1.0,
        1.0 if db_error else 0.0,
        1.0 if logic_config_error else 0.0,
        1.0 if async_order_error else 0.0,
        error_ratio,
    ]

    print(
        f"[State] service={service_name} max_duration={max_dur}ms "
        f"http_slow={http_slow} netem={netem_delay_active} "
        f"network_delay={network_delay} overload={overload_timeout} "
        f"db_error={db_error}"
    )

    if NOISE_LEVEL > 0:
        noise = np.random.normal(0, NOISE_LEVEL, len(state))
        state = np.clip(np.array(state) + noise, 0.0, 1.0).tolist()

    return state


# ──────────────────────────────────────────────────────────
# 7. A3 ACTION SELECTION
# ──────────────────────────────────────────────────────────
def select_action(state, a3_policy, actions):
    state_t = torch.tensor(state, dtype=torch.float)
    with torch.no_grad():
        probs      = a3_policy(state_t)
        action_idx = probs.argmax().item()
        action     = actions[action_idx]
        confidence = probs[action_idx].item() * 100
    return action, confidence, probs.tolist()


# ──────────────────────────────────────────────────────────
# 9. ACTION EXECUTOR
# ──────────────────────────────────────────────────────────
recovery_executor = RecoveryExecutor()

def get_container_name(service):
    return recovery_executor.container_name(service)

def execute(action, service, *, approved=False, context=None):
    """Execute and verify a recovery action.

    Returns a structured dictionary with executed, verified, mttr, message,
    and action-specific details.
    """
    print(f"\n  Executing {action} on {service}...")
    result = recovery_executor.execute(
        action,
        service,
        approved=approved,
        context=context or {},
    )
    print(f"  {result['message']}")
    raw_mttr = result.get("mttr")
    mttr_text = f"{float(raw_mttr):.2f}s" if raw_mttr is not None else "—"
    print(
        f"  executed={result.get('executed', False)} "
        f"verified={result.get('verified', False)} mttr={mttr_text}"
    )
    return result

# ──────────────────────────────────────────────────────────
# 10. DYNAMIC MONITOR
# ──────────────────────────────────────────────────────────
class ServiceMonitor:
    """
    Keeps per-service baseline information and persists calibrated thresholds.
    """

    BASELINE_WARMUP_TRACES = 20
    MIN_ABSOLUTE_JUMP      = 0.05
    RELATIVE_MULTIPLIER    = 1.50
    HEAL_COOLDOWN_SECONDS  = 120

    MAX_BASELINE_SCORE     = 0.05
    POLLUTED_BASELINE_MAX  = 0.05
    CLEAN_SCORE_MAX        = 0.01

    def __init__(self, baseline_store_path=BASELINE_STORE_PATH):
        self.baseline_store_path = baseline_store_path
        self.baselines       = {}
        self.baseline_counts = {}
        self.last_trace      = {}
        self.healed          = {}
        self.repeat_tracker  = {}
        self.warmup_scores   = defaultdict(list)
        self.thresholds      = {}
        self.load_baselines()

    def load_baselines(self):
        """Load saved baselines and repair invalid threshold values."""
        if not os.path.exists(self.baseline_store_path):
            print(f"No saved baseline file found: {self.baseline_store_path}")
            return

        repaired = False
        try:
            with open(self.baseline_store_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            services = payload.get("services", payload)
            loaded = 0

            for service, values in services.items():
                baseline = values.get("baseline")
                threshold = values.get("threshold")
                count = values.get("count", self.BASELINE_WARMUP_TRACES)

                if baseline is None:
                    continue

                baseline = float(baseline)
                threshold = float(threshold) if threshold is not None else None

                if not np.isfinite(baseline) or baseline < 0:
                    print(f"[{service}] ignoring invalid baseline: {baseline}")
                    continue

                minimum_threshold = max(baseline * 1.10, baseline + 1e-9)
                if threshold is None or not np.isfinite(threshold) or threshold <= baseline:
                    print(
                        f"[{service}] repaired invalid threshold: "
                        f"baseline={baseline:.6f}, old={threshold}, "
                        f"new={minimum_threshold:.6f}"
                    )
                    threshold = minimum_threshold
                    repaired = True

                self.baselines[service] = baseline
                self.thresholds[service] = threshold
                self.baseline_counts[service] = max(int(count), self.BASELINE_WARMUP_TRACES)
                loaded += 1

            print(f"Loaded {loaded} saved service baseline(s) from {self.baseline_store_path}")
            if repaired:
                self.save_baselines()
        except Exception as e:
            print(f"Could not load saved baselines: {e}")

    def save_baselines(self):
        """Persist only valid baseline/threshold pairs atomically."""
        services = {}

        for service, baseline in self.baselines.items():
            baseline = float(baseline)
            if not np.isfinite(baseline) or baseline < 0:
                continue

            threshold = self.thresholds.get(service)
            threshold = float(threshold) if threshold is not None else baseline * 1.10
            threshold = max(threshold, baseline * 1.10, baseline + 1e-9)
            self.thresholds[service] = threshold

            services[service] = {
                "baseline": baseline,
                "threshold": threshold,
                "count": int(self.baseline_counts.get(service, self.BASELINE_WARMUP_TRACES)),
                "updated_at": datetime.now().isoformat(),
            }

        payload = {"version": 1, "services": services}

        try:
            folder = os.path.dirname(self.baseline_store_path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            temp_path = self.baseline_store_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(temp_path, self.baseline_store_path)
        except Exception as e:
            print(f"Could not save baselines: {e}")

    def reset_baselines(self):
        """Clear in-memory and persisted calibration data."""
        self.baselines.clear()
        self.baseline_counts.clear()
        self.warmup_scores.clear()
        self.thresholds.clear()

        try:
            if os.path.exists(self.baseline_store_path):
                os.remove(self.baseline_store_path)
            print("Saved service baselines reset.")
        except Exception as e:
            print(f"Could not remove baseline file: {e}")

    def update_baseline(self, service, score, force=False):
        """Update baselines using healthy scores only.

        `force` is retained for compatibility, but it no longer permits
        anomalous scores to contaminate the healthy baseline.
        """
        score = float(score)

        if not np.isfinite(score) or score < 0:
            return False

        if score > self.MAX_BASELINE_SCORE:
            print(
                f"[{service}] baseline update rejected: score={score:.6f} "
                f"exceeds MAX_BASELINE_SCORE={self.MAX_BASELINE_SCORE:.6f}"
            )
            return False

        current_threshold = self.thresholds.get(service)
        if self.baseline_ready(service) and current_threshold is not None and score >= current_threshold:
            print(
                f"[{service}] anomaly excluded from baseline: "
                f"score={score:.6f}, threshold={current_threshold:.6f}"
            )
            return False

        self.baseline_counts[service] = self.baseline_counts.get(service, 0) + 1

        if not self.baseline_ready(service):
            self.warmup_scores[service].append(score)

        if service not in self.baselines:
            self.baselines[service] = score
        else:
            alpha = 0.05
            self.baselines[service] = (1.0 - alpha) * self.baselines[service] + alpha * score

        if self.baseline_ready(service):
            baseline = float(self.baselines[service])
            old_threshold = float(self.thresholds.get(service, baseline * 1.10))
            self.thresholds[service] = max(old_threshold, baseline * 1.10, baseline + 1e-9)
            self.save_baselines()

        return True

    def finalize_threshold(self, service):
        scores = np.asarray(self.warmup_scores.get(service, []), dtype=float)
        scores = scores[
            np.isfinite(scores)
            & (scores >= 0)
            & (scores <= self.MAX_BASELINE_SCORE)
        ]

        if len(scores) == 0:
            baseline = float(self.baselines.get(service, 0.0001))
            threshold = max(baseline * 1.10, baseline + 1e-9)
        elif len(scores) < 5:
            baseline = float(scores.mean())
            threshold = max(baseline * 1.10, baseline + 1e-9)
        else:
            baseline = float(scores.mean())
            percentile_threshold = float(np.percentile(scores, 99))
            threshold = max(
                percentile_threshold * 1.05,
                baseline * 1.10,
                baseline + 1e-9,
            )

        self.baselines[service] = baseline
        self.thresholds[service] = threshold
        self.baseline_counts[service] = max(
            self.baseline_counts.get(service, 0),
            self.BASELINE_WARMUP_TRACES,
        )
        self.save_baselines()

        print(
            f"  Baseline calibrated:"
            f"\n    Mean      : {baseline:.6f}"
            f"\n    Threshold : {threshold:.6f}"
        )

    def baseline_ready(self, service):
        return self.baseline_counts.get(service, 0) >= self.BASELINE_WARMUP_TRACES

    def get_threshold(self, service):
        if service not in self.thresholds:
            self.finalize_threshold(service)
        return self.thresholds[service]

    def recently_healed(self, service, trace_id):
        if service not in self.healed:
            return False

        last_time, last_trace = self.healed[service]

        # Never heal the same trace twice.
        if last_trace == trace_id:
            return True

        # Also avoid repeated actions shortly after a heal.
        return (time.time() - last_time) < self.HEAL_COOLDOWN_SECONDS

    def repeated_anomaly(self, service, score, tolerance=0.005, streak_needed=3):
        """
        Detect repeated near-identical anomaly scores.

        This usually means normal recurring traffic is being flagged due to stale
        or too-low baseline, not a real changing live fault.
        """
        last_score, streak = self.repeat_tracker.get(service, (None, 0))

        if last_score is not None and abs(score - last_score) < tolerance:
            streak += 1
        else:
            streak = 1

        self.repeat_tracker[service] = (score, streak)
        return streak >= streak_needed

    def mark_healed(self, service, trace_id):
        self.healed[service] = (time.time(), trace_id)


def scan_once(monitor, gnn_model, embedder, a3_policy, actions, llm, tokenizer, target_service=None):
    """Scan all services once for anomalies."""
    services = get_all_services()

    if not services:
        print("No services found in SkyWalking")
        return

    if target_service:
        services = {k: v for k, v in services.items() if target_service in k}

    print(f"\n  Monitoring {len(services)} services: {list(services.keys())}")

    for service_name, service_id in services.items():
        # 1. Get latest trace from SkyWalking.
        trace = get_latest_trace(service_id)
        if not trace:
            continue

        current_trace_id = trace["traceIds"][0]

        # 2. Skip same trace already processed in previous scan.
        if monitor.last_trace.get(service_name) == current_trace_id:
            continue
        monitor.last_trace[service_name] = current_trace_id

        # 3. Fetch spans.
        spans = get_spans(current_trace_id)
        if not spans:
            continue

        # 4. Score trace with GNN.
        score, root_cause = score_spans(spans, gnn_model, embedder)
        if score is None:
            continue

        # 5. Repair a polluted baseline.
        #    Example problem you saw:
        #      baseline=0.188772 while clean scores are around 0.0002.
        #    If a clean trace appears after an accidentally high baseline, reset it.
        if (monitor.baselines.get(service_name, 0.0) > monitor.POLLUTED_BASELINE_MAX
                and score < monitor.CLEAN_SCORE_MAX):
            print(
                f"  [{service_name:35s}] RESET BASELINE "
                f"old={monitor.baselines[service_name]:.6f} new={score:.6f}"
            )
            monitor.baselines[service_name] = score
            monitor.baseline_counts[service_name] = 1
            monitor.warmup_scores[service_name] = [score]
            monitor.thresholds.pop(service_name, None)
            monitor.save_baselines()
            continue

        # 6. First few clean traces are baseline only.
        #    Important: do NOT let old/high fault traces enter the baseline.
        if not monitor.baseline_ready(service_name):
            if score > monitor.MAX_BASELINE_SCORE:
                print(
                    f"  [{service_name:35s}] SKIP BASELINE "
                    f"score={score:.6f} looks too high"
                )
                continue

            monitor.update_baseline(service_name, score)
            count = monitor.baseline_counts[service_name]

            if count == monitor.BASELINE_WARMUP_TRACES:
                monitor.finalize_threshold(service_name)
                print(
                    f"  [{service_name:35s}] BASELINE READY "
                    f"baseline={monitor.baselines[service_name]:.6f} "
                    f"threshold={monitor.thresholds[service_name]:.6f} "
                    f"method=p99x1.05"
                )
            else:
                print(
                    f"  [{service_name:35s}] BASELINE "
                    f"score={score:.6f} "
                    f"count={count}/{monitor.BASELINE_WARMUP_TRACES}"
                )

            continue

        # 7. Compare against safer threshold.
        normal_score = monitor.baselines.get(service_name, score)
        threshold = monitor.get_threshold(service_name)

        if score < threshold:
            monitor.update_baseline(service_name, score)
            print(
                f"  [{service_name:35s}] NORMAL   "
                f"score={score:.6f} "
                f"baseline={normal_score:.6f} "
                f"threshold={threshold:.6f}"
            )
            continue

        # 8. It is above threshold, so treat it as an anomaly candidate.
        # Never update the healthy baseline using an anomalous score.

        print(f"\n  {'='*55}")
        print(f"  ANOMALY DETECTED: {service_name}")
        print(f"  {'='*55}")
        print(f"  Trace ID:   {current_trace_id}")
        print(f"  Score:      {normal_score:.6f} → {score:.6f}")
        print(f"  Threshold:  {threshold:.6f}")
        print(f"  Root cause: {root_cause}")
        print(f"  Duration:   {trace['duration']}ms")
        print(
            f"  Timestamp:  "
            f"{datetime.fromtimestamp(int(trace['start'])/1000).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # 8. Cooldown prevents repeated self-healing spam.
        if monitor.recently_healed(service_name, current_trace_id):
            print(f"Recently healed — skipping action (cooldown active)")
            continue

        # 9. Repeated same score usually means stale baseline / normal recurring trace.
        if monitor.repeated_anomaly(service_name, score):
            print(
                f"Same score repeating — likely normal recurring traffic "
                f"or stale baseline. Skipping action and letting baseline adapt."
            )
            continue

        # 10. Build RL state and select A3 action.
        state = build_state_from_trace(score, spans, service_name)
        print(f"  State dim: {len(state)}")

        action, confidence, probs = select_action(state, a3_policy, actions)

        print(f"\n  A3 Action: {action} ({confidence:.1f}% confidence)")
        print(f"  Probabilities:")
        for a, p in zip(actions, probs):
            bar = "█" * int(p * 20)
            print(f"    {a:15s}: {p*100:5.1f}%  {bar}")



        reason = LLM.ask_llm(action, state, score, normal_score, root_cause, service_name, llm, tokenizer)
        print(f"Reason: {reason}")

        # 12. Execute recovery.
        execute(action, service_name)
        monitor.mark_healed(service_name, current_trace_id)

        print(f"\n  Self-healing complete for {service_name} ")
        print(f"  {'='*55}")


# ──────────────────────────────────────────────────────────
# 11. MAIN
# ──────────────────────────────────────────────────────────
def main():
    
    
    parser = argparse.ArgumentParser(description="Dynamic self-healing agent")
    parser.add_argument("--once",    action="store_true",
                        help="Run single scan then exit")
    parser.add_argument("--service", type=str, default=None,
                        help="Monitor specific service only")
    parser.add_argument(
        "--reset-baselines",
        action="store_true",
        help="Delete saved per-service baselines and calibrate again",
    )
    args = parser.parse_args()

    print("="*60)
    print("DYNAMIC SELF-HEALING AGENT")
    print("="*60)
    print(f"SkyWalking: {SW_GRAPHQL}")
    print(f"Scan interval: {SCAN_INTERVAL}s")
    if args.service:
        print(f"Target service: {args.service}")
    print()

    # Load models
    gnn_model, embedder, a3_policy, actions, llm, tokenizer = load_models()  
    
    
    monitor = ServiceMonitor()
    if args.reset_baselines:
        monitor.reset_baselines()

    print("\n" + "="*60)
    print("Monitoring started. Press Ctrl+C to stop.")
    print("="*60)

    scan_count = 0
    try:
        while True:
            scan_count += 1
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[Scan #{scan_count} — {now}]")

            scan_once(monitor, gnn_model, embedder, a3_policy, actions, llm,tokenizer,
                     target_service=args.service)

            if args.once:
                print("\nSingle scan complete.")
                break

            print(f"\n  Next scan in {SCAN_INTERVAL}s...")
            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print(f"Total scans: {scan_count}")
        print(f"Services monitored: {list(monitor.baselines.keys())}")


if __name__ == "__main__":
    main()