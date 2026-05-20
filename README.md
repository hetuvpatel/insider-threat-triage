# insider-threat-triage
RL agent that monitors employee behavioral logs hourly, maintains a rolling risk score, and triages SOC alerts as Close / Investigate / Escalate. Trained on CERT Insider Threat v4.2 (2.38M rows, 0.18% malicious). Double DQN achieves 98.8% threat recall while auto-closing 67.8% of normal activity — directly solving SOC alert fatigue.
