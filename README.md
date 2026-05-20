# 🔐 Insider Threat Triage in Enterprise Systems Using Reinforcement Learning

> A reinforcement learning agent that monitors employee behavior across months of logs, maintains a rolling risk score, and makes triage decisions — *catching threats a rule-based system would entirely miss.*

**Course:** CP8319-CPS824 · Reinforcement Learning  
**Authors:** Hetu Patel (501215707) · Tenzin Tselha (501404675)

---

## 📌 Problem Statement

Security Operations Centers (SOCs) receive ~4,484 alerts daily yet cannot action **67%** of them. 83% of analysts report these are mostly false positives, while 91% worry about missing a real event (Vectra AI). Standard supervised ML predicts at each step but doesn't model how decisions affect future states — a fundamental mismatch with the sequential, slow-building nature of insider threats.

---

## 🧠 Approach: Reinforcement Learning as an MDP

We frame insider threat triage as a **Markov Decision Process**:

| Component | Detail |
|-----------|--------|
| **State** | 22-dimensional vector: 19 behavioral features + rolling risk score + previous action + steps since last escalation |
| **Actions** | `0 = Close` (dismiss), `1 = Investigate` (flag for human review), `2 = Escalate` (trigger incident response) |
| **Reward** | Asymmetric: +1.0 escalate malicious, −1.0 close malicious, −0.60 escalate normal, −0.05 investigate normal |
| **Transition** | Deterministic; each timestep = 1 hour of employee activity; risk score updates based on action taken |
| **Discount** | γ = 0.95 |

---

## 📊 Dataset: CERT Insider Threat v4.2

Carnegie Mellon University SEI standard benchmark.

- **2.38M** total hourly rows
- **1,000 employees** tracked over 17 months (Jan 2010 – May 2011)
- **0.18% malicious row rate** — extreme class imbalance
- **5 behavioral log sources:** logon, device (USB), file, email, HTTP

### Engineered Features (19 total)
Raw event counts, burst indicators (file ≥ 20, email ≥ 15, HTTP ≥ 30), per-user z-scores, after-hours flags, rare after-hours indicator, and interaction features (USB × after-hours, USB × file burst, etc.).

---

## ⚖️ Handling Extreme Class Imbalance (0.18%)

A naïve model predicting "normal" every time achieves 99.82% accuracy and catches **zero threats**. Four techniques combat this:

1. **Reward Asymmetry** — missing a threat (−1.0) penalized more than false alarm (−0.60)
2. **Buffer Pre-fill** — replay buffer pre-loaded with malicious transitions before training starts
3. **Episode Oversampling** — 70% of training episodes drawn from malicious users
4. **Dual Replay Buffer** — every training batch guaranteed 50% malicious transitions

---

## 🏗️ Models

### Model 1: Double Deep Q-Network (DQN)
- Off-policy learning with experience replay + target network
- Architecture: `22 → Linear(128) + LayerNorm + ReLU → Linear(128) + LayerNorm + ReLU → 3`
- Target network updated every N steps to stabilize training

### Model 2: SARSA (On-Policy)
- Same network architecture, but trains on the *actual* next action taken
- No replay buffer; single network; more conservative
- Update rule: `Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') − Q(s,a)]`

---

## 📈 Results

Evaluated on held-out 20% test set — 30,000 rows, 412 truly malicious.

| Metric | Double DQN | SARSA |
|--------|-----------|-------|
| **Recall (escalation)** | **98.8%** ✅ | 83.2% |
| Miss Rate | 1.2% | **0.0%** ✅ |
| Escalation Precision | 14.4% | **55.7%** ✅ |
| FP Escalation Rate | 8.1% | **1.3%** ✅ |
| **Normal Close Rate** | **67.8%** ✅ | 32.9% ❌ |
| Loss Stability | **Stable** ✅ | Diverges ❌ |

**Verdict: Double DQN wins for the SOC use case.**  
SARSA achieves 0 missed threats but routes 67% of normal activity to "Investigate" — recreating the alert fatigue problem. DQN closes 67.8% of normal alerts automatically while catching 98.8% of real threats.

**Rule-based baseline recall: 0.0% → DQN: 98.8% — RL necessity proven empirically.**

---

## 🗂️ Repository Structure

```
insider-threat-rl/
│
├── rl_code.py                  # Full training pipeline (DQN + SARSA)
│   ├── Data loading & feature engineering (CERT v4.2)
│   ├── CertTriageEnv           # Custom Gym-compatible RL environment
│   ├── DoubleDQNAgent          # Double DQN with dual replay buffer
│   ├── SARSAAgent              # On-policy SARSA agent
│   └── Evaluation & visualization
│
├── README.md
└── RL_Final_Presentation.pdf   # Full slide deck
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch numpy pandas scikit-learn matplotlib gymnasium
```

### Dataset
Request access to **CERT Insider Threat Dataset v4.2** from Carnegie Mellon University SEI:  
https://doi.org/10.1184/R1/12841247.v1

Place the extracted `r4.2/` folder and `answers/` folder at your data path.

### Running on Google Colab (recommended)
The code is written for Google Colab with Google Drive mounting. Update `DATA_PATH` and `ANSWERS_PATH` to point to your dataset location:

```python
DATA_PATH = '/content/r4.2'         # Update to your path
ANSWERS_PATH = '/content/answers'   # Update to your path
```

Then run all cells sequentially — feature engineering, environment setup, DQN training, SARSA training, and evaluation.

---

## ⚠️ Limitations & Future Work

| Limitation | Proposed Fix |
|------------|-------------|
| Markov approximation (EMA risk score ≠ full history) | Replace feedforward DQN with LSTM-DQN |
| DQN loss drift after episode 200 | Add learning rate decay schedule |
| SARSA TD loss divergence | Add experience replay / target network to SARSA |
| 3 employees still over-escalated at 55–75% | Per-employee fine-tuning or role-context features |

---

## 📚 Citation

```bibtex
@dataset{lindauer2020cert,
  author    = {Lindauer, B.},
  title     = {Insider Threat Test Dataset v1},
  year      = {2020},
  publisher = {Carnegie Mellon University},
  doi       = {10.1184/R1/12841247.v1}
}
```

---

## 📄 License

This project is for academic purposes (CP8319-CPS824). Dataset usage governed by CMU SEI terms.
