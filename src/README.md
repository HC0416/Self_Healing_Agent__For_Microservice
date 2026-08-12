# Learning-Based Self-Healing Agents for Fault Detection and Recovery in Microservice Systems

The project proposes an AI-driven self-healing framework that combines graph neural network (GGNN) anomaly detection with reinforcement learning-based recovery to automatically detect, diagnose, recover, and explain failures in cloud-native microservice systems.

The framework is evaluated using the Train-Ticket microservice benchmark together with Apache SkyWalking for distributed tracing.

---

## Features

- Distributed trace collection using Apache SkyWalking
- GGNN + Deep SVDD anomaly detection
- Attention-based root cause localisation
- Reinforcement learning recovery policy (REINFORCE)
- Adaptive recovery execution
- Automatic recovery verification
- Local LLM-based recovery explanation (Qwen2.5)
- Live monitoring dashboard

---

## Repository Structure

```
src/
 ├── pipeline/           # Offline training pipeline
 ├── self_healing_agent/ # Online monitoring and recovery framework
 ├── train-ticket/       # Train-Ticket benchmark
 └── requirements.txt
```

---

## System Architecture

The proposed framework consists of six major components:

1. Distributed Trace Collection
2. GGNN-Based Anomaly Detection
3. Reinforcement Learning Recovery Policy
4. Adaptive Recovery Executor
5. Recovery Verification
6. LLM-Based Recovery Explanation

The workflow is:

```
User Request
      │
      ▼
Apache SkyWalking
      │
      ▼
Trace Graph Construction
      │
      ▼
GGNN + Deep SVDD
      │
      ▼
Root Cause Localisation
      │
      ▼
RL Recovery Policy
      │
      ▼
Adaptive Recovery Executor
      │
      ▼
Recovery Verification
      │
      ▼
LLM Explanation
```

---

## Requirements

- Python 3.10+
- Docker
- Docker Compose
- Apache SkyWalking
- Train-Ticket benchmark
- CUDA (optional)

Main Python libraries include:

- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- Docker SDK
- Requests
- Transformers

Install dependencies using

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Start Train-Ticket

```bash
docker compose up -d
```

### 2. Start SkyWalking

Ensure both the OAP server and UI are running.

### 3. Run the Self-Healing Agent

```bash
python monitor.py
```

The framework continuously:

- collects distributed traces
- detects anomalies
- localises root causes
- predicts recovery actions
- executes recovery
- verifies recovery
- generates LLM explanations

---

## Recovery Actions

The reinforcement learning policy predicts one of the following actions:

- Restart
- Scale Up
- Reroute
- Rollback*

\* Rollback requires manual approval before execution.

---

## Evaluation

Offline evaluation compares:

- Isolation Forest
- GCN + Deep SVDD
- GGNN (mixed training)
- GGNN (normal-only)

Online evaluation includes:

- CPU overload
- Network delay
- Service crash
- Database failure

Metrics:

- Precision
- Recall
- F1-score
- AUC-ROC
- Detection Success Rate
- Recovery Success Rate
- Detection Latency
- MTTR

---

## Technologies

- Python
- PyTorch
- PyTorch Geometric
- Apache SkyWalking
- Docker
- Train-Ticket
- Qwen2.5
- Graph Neural Networks
- Deep SVDD
- Reinforcement Learning

---

## Experimental Results

The proposed framework was evaluated on the Train-Ticket microservice benchmark.

### Offline Anomaly Detection

| Model | AUC | Precision | Recall | F1 |
|------|----:|----------:|-------:|---:|
| Isolation Forest | 0.917 | 0.917 | 0.846 | 0.880 |
| GCN + Deep SVDD | 0.854 | 0.678 | 0.895 | 0.772 |
| GGNN (Mixed Training) | 0.869 | 0.756 | 0.913 | 0.827 |
| **GGNN (Normal-Only)** | **0.947** | **0.968** | **0.749** | **0.845** |

### Reinforcement Learning Recovery

| Method | Accuracy |
|---------|---------:|
| Rule-Based | 87.6% |
| REINFORCE | 91.9% |
| **Supervised Pretraining + REINFORCE** | **96.5%** |

### Online Evaluation

The deployed framework successfully detected and recovered all evaluated fault scenarios.

| Fault | Detection | Recovery |
|--------|:---------:|:--------:|
| CPU Overload | 10/10 |  10/10 |
| Network Delay | 10/10 |  10/10 |
| Service Crash | 10/10 |  10/10 |
| Database Error |  10/10 |  10/10 |

Overall:

- 100% detection success rate
- 100% recovery success rate
- Detection latency: **1.00–4.61 s**
- MTTR: **4.05–18.19 s**

