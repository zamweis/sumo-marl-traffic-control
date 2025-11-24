# Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control in SUMO

A complete framework for training and evaluating **multi-agent reinforcement learning (MARL)** systems for adaptive traffic signal control in realistic urban environments, using the **SUMO** microscopic traffic simulator.

This repository implements the full system developed in my Bachelor thesis **“Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control in SUMO”** (2025), including network preprocessing, agent training, evaluation, and reproducible experiments.

## 📋 Overview

This project provides an end-to-end pipeline for learning adaptive traffic signal control policies using multi-agent PPO. Key components:

- **Multi-Agent RL Training**  
  Each traffic light (TLS) is controlled by an independent RL agent using PPO (Stable-Baselines3).

- **Real-World Network Integration**  
  Based on an OpenStreetMap (OSM) extract of Karlsruhe, Germany, including 17 signalized intersections.

- **Multiple Reward Functions**  
  Evaluate different policy objectives:
  - waiting time
  - queue length
  - emissions
  - combined real-world metrics

- **Automated Network Processing**  
  Includes several scripts to repair, validate, and clean OSM-imported networks.

- **Comprehensive Evaluation**  
  Systematic benchmarking against fixed-time and actuated baselines.

## 🏗️ Architecture

```
SUMO Simulation ↔ TraCI Interface ↔ SUMO-RL Environment ↔ Stable-Baselines3 (PPO)
                          ↑
                 Multi-Agent Controller
                          ↑
               17 Independent RL Agents
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- SUMO 1.18.0 or newer
- 16GB RAM recommended

### Installation

1. **Install SUMO**
   ```bash
   sudo apt-get install sumo sumo-tools sumo-doc
   ```

2. **Set the SUMO_HOME environment variable**
   ```bash
   export SUMO_HOME="/path/to/sumo"
   ```

3. **Clone this repository and install dependencies**
   ```bash
   git clone https://github.com/zamweis/sumo-marl-traffic-control.git
   cd sumo-marl-traffic-control
   pip install -r requirements.txt
   ```
4.  **Install sumo-rl**

    clone sumo-rl repository and add to path

5. **Implement emissions in sumo-rl**
   
    in sumo_rl/environment/env.py modify  _compute_info(self)
    ```
    def _compute_info(self):
        info = {"step": self.sim_step}
        if self.add_system_info:
            info.update(self._get_system_info())

            # --- NEU: CO₂ Emissionen ---
            lanes = []
            for ts in self.traffic_signals.values():
                lanes.extend(ts.lanes)   # <<--- wichtig!
            if lanes:
                total_co2 = sum(self.sumo.lane.getCO2Emission(lane) for lane in lanes)
                n_veh = sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in lanes)
                mean_co2 = total_co2 / max(1, n_veh)
                info["system_mean_co2"] = mean_co2

        if self.add_per_agent_info:
            info.update(self._get_per_agent_info())
        self.metrics.append(info.copy())
        return info
     ```

## ▶️ Basic Usage

### **1. Prepare or repair the SUMO network**
Use one of the provided repair scripts:

```bash
python scripts/repair_net.py
python scripts/check_tls_consistency.py
.
.
.
python scripts/find_valid_tls.py
```
Some networks need manual deletion of railway tls.

### **2. Train an RL model**
```bash
python train.py
```

### **3. Continue training a saved model**
```bash
python continuetrain.py
```

### **4. Evaluate trained models**
```bash
python evaluate.py
```

Evaluation results are saved in JSON and CSV format.

## 📂 Repository Structure

```
sumo-marl-traffic-control/
├── scripts/                     # Network processing & SUMO/OSM repair tools
│   ├── check_tls_consistency.py
│   ├── fix_requests.py
│   ├── repair_net.py
│   └── find_valid_tls.py
│
├── latex/                       # Thesis (LaTeX source)
│   └── ...                     
│
├── src/
│   ├── diffwaitingtime/         # Reward: diff-waiting-time (training + evaluation)
│   │   ├── train.py
│   │   ├── continuetrain.py
│   │   ├── evaluate.py
│   │   ├── evaluation/          # Auto-created evaluation outputs
│   │   └── runs/                # Auto-created model checkpoints + logs
│   │
│   ├── realworld/               # Reward: real-world metric combination
│   │   ├── train.py
│   │   ├── continuetrain.py
│   │   ├── evaluate.py
│   │   ├── evaluation/
│   │   └── runs/
│   │
│   ├── emissions/               # Reward: CO₂ emission minimization
│   │   ├── train.py
│   │   ├── continuetrain.py
│   │   ├── evaluate.py
│   │   ├── evaluation/
│   │   └── runs/
│   │
│   ├── queue/                   # Reward: queue-length reduction
│   │   ├── train.py
│   │   ├── continuetrain.py
│   │   ├── evaluate.py
│   │   ├── evaluation/
│   │   └── runs/
│
├── data/                        # SUMO network, traffic flows & config files
│   ├── map.net.xml
│   ├── flows_*.rou.xml
│   └── sumoconfig.sumocfg
│
└── runs/ (optional)             # Only used if models are moved here manually
```

## 🎯 Key Features

### Reward Functions
- **Diff-Waiting-Time**: Minimize cumulative waiting time difference  
- **Queue**: Minimize the number of stopped vehicles  
- **Real-World**: Weighted combination of speed, queue length, waiting time  
- **Emissions**: Reduce CO₂ emissions while maintaining traffic flow  

### Network Processing
- Automated OSM → SUMO conversion  
- TLS consistency checking and repair  
- Automatic request index fixing  
- Identification of RL-compatible TLS  
- Validation of lane and edge structures  

### RL Training
- Multi-agent PPO (Stable-Baselines3)  
- Scenario-based curriculum learning  
- TensorBoard logging  
- Periodic checkpoints and best model selection  

## 📚 Citation

If you use this repository in your research, please cite:

```
@thesis{weiler2025marl,
  title={Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control in SUMO},
  author={Weiler, Sam},
  year={2025},
  institution={Hochschule Karlsruhe}
}
```

## 🤝 Contributing

Contributions are welcome!  
Feel free to open issues or submit pull requests.

## 📄 License

Licensed under the MIT License.  
See the `LICENSE` file for details.

## 🙏 Acknowledgments

- Prof. Dr. Patrick Baier  
- Prof. Dr. Heiko Körner  
- SUMO development team  
- Lucas Alegre (sumo-rl)  
- Stable-Baselines3 team
