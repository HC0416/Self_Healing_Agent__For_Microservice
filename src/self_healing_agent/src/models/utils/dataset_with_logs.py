"""
utils/dataset_with_logs.py
==========================
Loads DeepTraLog GraphData (.jsons) combined with alllog.log
for full trace+log TEG (Trace Event Graph) representation.

Pipeline:
  1. Collect all TraceIds from GraphData (fast pass)
  2. Load alllog.log → filter ERROR/WARN + needed traces only
  3. Load GraphData process*.jsons → traces with labels
  4. For each trace → find matching logs from alllog
  5. Add log nodes to graph (ERROR/WARN only)
  6. Return PyG Data object with combined TEG

Node features (306-dim):
  GloVe 300-dim + 6 structural features
"""

import os, json, re, random
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, Subset

# ── Constants ─────────────────────────────────────────────────
N_SERVICES    = 35
NODE_FEAT_DIM = 306   # GloVe 300-dim + 6 structural features

# ── Log line regex ────────────────────────────────────────────
LOG_PATTERN = re.compile(
    r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+'
    r'\s+\[SW_CTX:\[([^,]+),[^,]+,([^,]+),([^,]+),'
    r'[^\]]+\]\]\s+\[[^\]]+\]\s+(\w+)\s+\S+\s+-(.+)$'
)

LOG_LEVEL_MAP = {
    'ERROR': 1.0,
    'WARN':  0.7,
    'INFO':  0.3,
    'DEBUG': 0.1,
}

EDGE_PARENT_CHILD = 0
EDGE_SEQUENCE     = 1

STOP_WORDS = {
    'a','an','the','is','it','in','on','at','to','for',
    'of','and','or','with','from','by','be','has','have',
    'was','were','are','this','that','get','post','put',
    'api','v1','http',
}


# ══════════════════════════════════════════════════════════════
# GloVe Embedder
# ══════════════════════════════════════════════════════════════

class GloVeEmbedder:
    def __init__(self, glove_path):
        print(f"Loading GloVe from {glove_path}...")
        self.vectors = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word  = parts[0]
                vec   = np.array(parts[1:], dtype=np.float32)
                self.vectors[word] = vec
        print(f"Loaded {len(self.vectors):,} word vectors")
        self.dim = 300

    def embed(self, text: str) -> np.ndarray:
        text  = text.lower()
        text  = re.sub(r'[^a-z\s]', ' ', text)
        words = [w for w in text.split()
                 if w and w not in STOP_WORDS]
        vecs  = [self.vectors[w] for w in words
                 if w in self.vectors]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)


# ══════════════════════════════════════════════════════════════
# Load alllog — ERROR/WARN only + needed traces only
# ══════════════════════════════════════════════════════════════

def load_alllog(alllog_path: str,
               needed_trace_ids: set = None) -> dict:
    """
    Load alllog.log filtered to:
      1. ERROR/WARN only  → removes INFO noise, saves RAM
      2. needed_trace_ids → only loads logs for known traces

    Returns: dict mapping trace_id → list of log dicts
    """
    print(f"\nLoading alllog from {alllog_path}...")
    if needed_trace_ids:
        print(f"  Filtering to {len(needed_trace_ids):,} "
              f"known traces + ERROR/WARN only")

    logs_by_trace  = defaultdict(list)
    total          = 0
    filtered_level = 0
    filtered_trace = 0
    skipped        = 0

    with open(alllog_path, 'r',
              encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = LOG_PATTERN.match(line.strip())
            if not m:
                skipped += 1
                continue

            level    = m.group(4).strip()
            trace_id = m.group(2).strip()

            # Filter 1: ERROR/WARN only
            if level not in ('ERROR', 'WARN'):
                filtered_level += 1
                continue

            # Filter 2: only traces we need
            if needed_trace_ids and \
               trace_id not in needed_trace_ids:
                filtered_trace += 1
                continue

            logs_by_trace[trace_id].append({
                'service': m.group(1).strip(),
                'span_id': m.group(3).strip(),
                'level':   level,
                'message': m.group(5).strip(),
            })
            total += 1

    print(f"  Loaded {total:,} ERROR/WARN lines "
          f"across {len(logs_by_trace):,} traces")
    print(f"  Skipped {filtered_level:,} INFO/DEBUG lines")
    print(f"  Skipped {filtered_trace:,} unknown trace lines")
    return logs_by_trace


# ══════════════════════════════════════════════════════════════
# Build TEG
# ══════════════════════════════════════════════════════════════

def node_info_to_features(node_info: list) -> list:
    duration   = float(node_info[0]) / 10000.0
    service_id = int(node_info[1])
    parent     = int(node_info[2])
    type_id    = float(node_info[5]) / 10.0
    is_root    = 1.0 if parent == -1 else 0.0
    svc_onehot = [0.0] * N_SERVICES
    if 0 <= service_id < N_SERVICES:
        svc_onehot[service_id] = 1.0
    return [duration, is_root] + svc_onehot + [type_id]


def build_teg(graph_dict: dict,
              trace_logs: list,
              embedder: GloVeEmbedder,
              max_spans: int = 50) -> Data:
    """Build TEG from GraphData graph + matching logs."""
    node_info  = graph_dict['node_info']
    edges      = graph_dict.get('edge_index', [])
    trace_bool = graph_dict.get('trace_bool', True)
    label      = 0 if trace_bool else 1

    # Limit spans to prevent huge graphs
    if len(node_info) > max_spans:
        node_info = node_info[:max_spans]
        edges = [e for e in edges
                 if len(e) >= 2
                 and e[0] < max_spans
                 and e[1] < max_spans]

    n_spans = len(node_info)

    durations      = [float(n[0]) for n in node_info]
    mean_dur       = max(np.mean(durations), 1.0)
    children_count = defaultdict(int)
    for edge in edges:
        if len(edge) >= 2:
            children_count[edge[0]] += 1
    n_error_spans = sum(
        1 for n in node_info if float(n[0]) > 5000.0)
    n_error_ratio = n_error_spans / max(n_spans, 1)

    nodes     = []
    node_meta = []
    edges_src = []
    edges_dst = []
    edge_type = []

    # ── Span nodes ────────────────────────────────────────────
    for i, ni in enumerate(node_info):
        svc_id   = int(ni[1])
        type_id  = int(ni[5])
        span_txt = f"service {svc_id} type {type_id}"
        glove    = embedder.embed(span_txt)

        is_error   = 1.0 if float(ni[0]) > 5000.0 else 0.0
        n_children = float(children_count.get(i, 0))
        rel_dur    = float(ni[0]) / mean_dur
        extra = np.array([
            is_error, 0.0, 0.0,
            n_children, rel_dur, n_error_ratio
        ], dtype=np.float32)

        nodes.append(np.concatenate([glove, extra]))
        node_meta.append({
            'type': 'span', 'service': svc_id, 'index': i})

    # ── Span→span edges ───────────────────────────────────────
    for edge in edges:
        if len(edge) >= 2:
            edges_src.append(edge[0])
            edges_dst.append(edge[1])
            edge_type.append(EDGE_PARENT_CHILD)

    # ── Log nodes (ERROR/WARN only, max 10) ───────────────────
    for log in trace_logs[:10]:
        glove    = embedder.embed(log['message'])
        is_error = 1.0 if log['level'] == 'ERROR' else 0.0
        log_lvl  = LOG_LEVEL_MAP.get(log['level'], 0.0)
        extra    = np.array([
            is_error, log_lvl, 0.0,
            0.0, 0.0, n_error_ratio
        ], dtype=np.float32)

        log_idx = len(nodes)
        nodes.append(np.concatenate([glove, extra]))
        node_meta.append({
            'type':    'log',
            'level':   log['level'],
            'message': log['message'],
            'service': log['service'],
        })
        # Connect log to root span
        edges_src.append(0)
        edges_dst.append(log_idx)
        edge_type.append(EDGE_SEQUENCE)

    # ── Build PyG Data ────────────────────────────────────────
    x = torch.tensor(np.stack(nodes), dtype=torch.float)

    if edges_src:
        edge_index = torch.tensor(
            [edges_src, edges_dst], dtype=torch.long)
        edge_attr  = torch.tensor(
            [[t] for t in edge_type], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 1), dtype=torch.long)

    y    = torch.tensor([label], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, y=y)
    data.node_meta  = node_meta
    data.trace_id   = graph_dict.get('trace_id', '')
    data.error_type = graph_dict.get('error_trace_type',
                                     'normal')
    return data


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class DeepTraLogDatasetWithLogs(Dataset):
    """
    Full DeepTraLog dataset: GraphData + alllog.log
    Loads ERROR/WARN logs only to save RAM.
    """

    def __init__(self, graph_dir: str,
                 alllog_path: str,
                 embedder: GloVeEmbedder,
                 max_per_file: int = None):
        self.graphs = []
        self._load(graph_dir, alllog_path,
                   embedder, max_per_file)

    def _load(self, graph_dir, alllog_path,
              embedder, max_per_file):
        import glob
        jsons_files = sorted(
            glob.glob(os.path.join(
                graph_dir, 'process*.jsons')))

        if not jsons_files:
            raise FileNotFoundError(
                f"No process*.jsons in {graph_dir}")

        print(f"Found {len(jsons_files)} .jsons files")

        # Pass 1: collect all TraceIds from GraphData
        print("Pass 1: Collecting trace IDs...")
        needed_ids = set()
        for fpath in jsons_files:
            with open(fpath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        gdict = json.loads(line)
                        tid   = gdict.get('trace_id', '')
                        if tid:
                            needed_ids.add(tid)
                    except:
                        continue
        print(f"Found {len(needed_ids):,} unique trace IDs")

        # Load alllog filtered
        logs_by_trace = load_alllog(
            alllog_path, needed_trace_ids=needed_ids)

        # Pass 2: build TEG graphs
        print("\nPass 2: Building TEG graphs...")
        total, skipped, matched = 0, 0, 0

        for fpath in jsons_files:
            fname = os.path.basename(fpath)
            count = 0
            with open(fpath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        gdict = json.loads(line)
                        tid   = gdict.get('trace_id', '')
                        logs  = logs_by_trace.get(tid, [])
                        if logs:
                            matched += 1
                        data = build_teg(gdict, logs, embedder)
                        self.graphs.append(data)
                        count += 1
                        total += 1
                    except Exception:
                        skipped += 1
                        continue
                    if max_per_file and count >= max_per_file:
                        break
            print(f"  {fname}: {count} graphs")

        n_normal = sum(1 for g in self.graphs
                       if g.y.item() == 0)
        n_anom   = sum(1 for g in self.graphs
                       if g.y.item() == 1)
        print(f"\nTotal: {total} graphs "
              f"(normal={n_normal}, anomalous={n_anom})")
        print(f"Graphs with logs: {matched} "
              f"({matched/total*100:.1f}%)")
        print(f"Node feature dim: {NODE_FEAT_DIM}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# ══════════════════════════════════════════════════════════════
# Main loader
# ══════════════════════════════════════════════════════════════

def load_data_with_logs(graph_dir: str,
                        alllog_path: str,
                        glove_path: str,
                        batch_size: int = 32,
                        seed: int = 42,
                        max_per_file: int = None,
                        normal_only_train: bool = True):
    """
    Load GraphData + alllog and return DataLoaders.
    GloVe and alllog are loaded once and reused.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    embedder = GloVeEmbedder(glove_path)
    dataset  = DeepTraLogDatasetWithLogs(
        graph_dir, alllog_path, embedder, max_per_file)

    n   = len(dataset)
    idx = list(range(n))
    random.shuffle(idx)

    if normal_only_train:
        normal_idx = [i for i in idx
                      if dataset[i].y.item() == 0]
        anom_idx   = [i for i in idx
                      if dataset[i].y.item() == 1]

        n_train = int(len(normal_idx) * 0.8)
        n_val   = int(len(normal_idx) * 0.1)

        train_idx = normal_idx[:n_train]
        val_idx   = (normal_idx[n_train:n_train+n_val]
                     + anom_idx[:len(anom_idx)//2])
        test_idx  = (normal_idx[n_train+n_val:]
                     + anom_idx[len(anom_idx)//2:])

        print(f"\nTrain: {len(train_idx)} (normal only)")
        print(f"Val:   {len(val_idx)}")
        print(f"Test:  {len(test_idx)}")
    else:
        t = int(n * 0.7)
        v = int(n * 0.85)
        train_idx = idx[:t]
        val_idx   = idx[t:v]
        test_idx  = idx[v:]
        print(f"\nTrain: {len(train_idx)} (mixed)")
        print(f"Val:   {len(val_idx)}")
        print(f"Test:  {len(test_idx)}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size)
    test_loader  = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size)

    return (train_loader, val_loader, test_loader,
            dataset, embedder)