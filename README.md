# Reinforcement Learning for Bi-Dexterous Manipulation of a Rubik’s Cube

This repository explores **reinforcement learning for bi-dexterous manipulation** using two Shadow Hands in a MuJoCo simulation environment. The goal is to train agents that can maintain a stable grasp on a Rubik’s Cube and execute sequential face rotations through contact-rich, coordinated control.

The project investigates both **single-agent PPO** and **multi-agent PPO (MAPPO-style)** formulations for a task that is significantly more challenging than simple grasping: it requires long-horizon planning, stable contact, role specialization between hands, and precise rotational control.

---

## Motivation

Dexterous manipulation is a key requirement for general-purpose robotics. While prior work has demonstrated impressive single-hand dexterity, true bimanual manipulation remains difficult because it introduces:

- Higher-dimensional action spaces
- Complex multi-contact dynamics
- Long-horizon credit assignment
- Coordination and role specialization between two hands
- Stability challenges under gravity and contact-rich interaction

A Rubik’s Cube provides a useful benchmark-style task because it is:

- **Contact-rich**: fingers must apply controlled forces without destabilizing the cube.
- **Long-horizon**: successful manipulation may require multiple sequential rotations.
- **Highly coordinated**: both hands must maintain grasp stability while one or both hands contribute to rotation.
- **Physically challenging**: freejoint settings introduce gravity, slipping, crushing, and regrasping failures.

---

## Research Context

This work is motivated by two related directions:

1. **Single-hand dexterous manipulation**
   - Akkaya et al., *Solving Rubik’s Cube with a Robot Hand*, 2019.
   - Demonstrates the feasibility of complex manipulation with a single robotic hand.

2. **Bimanual dexterous manipulation**
   - Chen et al., *Towards Human-Level Bimanual Dexterous Manipulation with Reinforcement Learning (Bi-DexHands)*, 2023.
   - Provides bimanual RL benchmarks, but many tasks are simpler than contact-rich sequential cube manipulation.

This project positions the Rubik’s Cube task as a harder bimanual manipulation problem where the agent must learn grasping, force control, face rotation, and sequence completion together.

---

## Environment Design

The simulation is built using **MuJoCo** with two Shadow Hands and a Rubik’s Cube model.

### Robot Hands

Each Shadow Hand includes:

- Wrist and palm body
- Five independently actuated fingers
- Approximately **26 DOF per hand**
- Contact sensors and joint state information used for reward and state construction

### Rubik’s Cube

Two cube variants are considered:

| Cube Variant | Description | Purpose |
|---|---|---|
| **Hinged cube** | Cube face rotations are modeled with hinge joints. | Easier controlled setting for learning face rotation. |
| **Freejoint cube** | Cube moves freely under physics and gravity. | Harder setting with realistic stability and contact challenges. |

The hinged cube contains:

- **27 rigid bodies**
- **6 hinge joints**, one for each cube face rotation

---

## Problem Formulation

The task is formulated as a **continuous-state, continuous-action Markov Decision Process (MDP)**.

### Objective

The policy is trained to:

1. Maintain a stable bimanual grasp.
2. Rotate a target cube face in the correct direction.
3. Complete one or more face rotations in sequence.
4. Avoid unstable behavior such as slipping, excessive palm contact, crushing, or pushing the cube outside the workspace.

### State Space

The state representation includes information such as:

- Hand joint positions and velocities
- Cube pose and velocity
- Face rotation angles
- Fingertip and palm contact forces
- Target face and rotation direction
- Progress through the rotation sequence
- Previous actions or control history, where applicable

### Action Space

The action space is continuous and controls the actuators of both Shadow Hands. Depending on the environment variant, the action may include:

- Left-hand actuator commands
- Right-hand actuator commands
- Cube/face actuation components for hinged-cube experiments

### Transition Dynamics

State transitions are governed by MuJoCo rigid-body dynamics, including:

- Contact forces
- Friction
- Joint constraints
- Gravity
- Multi-body interactions between fingers, palms, and cube bodies

### Termination Conditions

Episodes may terminate when:

- The target rotation sequence is completed.
- The cube leaves the workspace.
- The grasp becomes unstable.
- The maximum episode length is reached.
- Physical constraints are violated.

---

## Reward Design

The reward function combines grasp stability, safe contact, workspace stability, and rotation progress.

### Stability Rewards and Penalties

| Component | Purpose |
|---|---|
| **Grasp stability reward** | Encourages fingertip normal contact forces to stay within a desirable range. |
| **Palm contact penalty** | Penalizes excessive palm-cube collisions. |
| **Workspace stability penalty** | Penalizes displacement of the cube away from its initial or desired workspace center. |

### Rotation Rewards

| Component | Purpose |
|---|---|
| **Rotation initiation reward** | Gives a small reward when the target face begins rotating in the correct direction. |
| **Rotation progress reward** | Rewards incremental positive angular progress. |
| **Rotation completion reward** | Gives a larger reward when the face reaches the target rotation, typically 90°. |
| **Sequence completion reward** | Gives a high terminal reward when the full rotation sequence is completed. |

This reward structure is important because the task involves sparse long-horizon goals but requires dense feedback for grasping and contact stabilization.

---

## Algorithms

### Proximal Policy Optimization (PPO)

The main training algorithm is **PPO** with an actor-critic architecture.

Key design choices:

- Shared MLP backbone for policy and value representations
- Separate policy and critic heads
- Gaussian continuous action policy
- Learned, state-independent log standard deviation
- Clipped PPO objective
- Value function clipping
- Generalized Advantage Estimation (GAE)

The policy outputs the mean of a multivariate Gaussian action distribution with diagonal covariance:

```text
a ~ N(mu_theta(s), diag(sigma^2))
```

where the log standard deviation is learned to prevent premature variance collapse.

### PPO Stability Mechanisms

Several stability mechanisms are used to improve learning:

- **Max value clipping** for the critic loss
- **KL-based adaptive learning-rate scheduling**
- **Delayed policy updates** until enough completed episodes and buffer samples are available
- **Vectorized PPO memory and optimization**
- Training with up to **256 parallel environments**

### Multi-Agent PPO / MAPPO-style Setup

A multi-agent variant is also explored, where each hand is treated as a separate agent.

Each agent receives:

- The full environment state
- The same shared reward signal
- Its own policy parameters
- Its own critic parameters
- Its own rollout buffer

This setup investigates whether separate policies for each hand can encourage coordination and role specialization.

---

## Experiments

The project evaluates PPO and MAPPO-style approaches across hinged-cube and freejoint-cube settings.

### Hinged Cube Experiments

The hinged cube experiments test whether the agent can learn controlled face rotations under a more structured cube model.

Tested rotation sequences include:

| Experiment | Rotation Sequence |
|---|---|
| Single-face rotation | `blue_clock` |
| Two-face rotation | `blue_clock`, `white_anti_clock` |
| Two-face rotation | `green_anti_clock`, `red_clock` |
| Three-face rotation | `blue_clock`, `white_anti_clock`, `green_anti_clock` |
| Generalization test | `blue_clock`, `white_anti_clock` |

### Freejoint Cube Experiments

The freejoint cube setting is more physically difficult because the cube is no longer constrained by face hinges alone. Under gravity, the agent must maintain full-object stability while manipulating the cube.

Observed failure modes include:

- Excessive fingertip force causing deformation or unstable contact
- Cube being pushed into the palm region and crushed
- One hand lifting or isolating the cube away from the workspace center
- Inconsistent regrasp attempts leading to loss of control

---

## Key Findings

- PPO can learn meaningful bi-dexterous control behavior in the hinged-cube setting.
- Reward shaping is critical for stable grasping and sequential rotation.
- Long-horizon credit assignment remains a major challenge.
- Emergent coordination and role specialization can appear between the two hands.
- PPO with stability mechanisms is more effective than naive PPO updates.
- MAPPO-style independent hand policies are not always better for fine-grained dexterous manipulation.
- Freejoint cube manipulation remains fragile because of complex physical interactions, gravity, and contact instability.

---

## Repository Structure

```text
.
├── assets/                     # Meshes, textures, and supporting simulation assets
├── environment/                # Rubik's Cube manipulation environments
├── logs/                       # Training logs
├── mujoco_trials/              # MuJoCo experiments and trial scripts
├── plots/                      # Training curves and result visualizations
├── primitive_controller/       # PPO and control-related implementations
├── saved_models/               # Saved policy checkpoints
├── shadow_hand/                # Shadow Hand model files and utilities
├── utils/                      # Utility functions
├── xmls/                       # MuJoCo XML model definitions
├── evaluate_ppo.py             # PPO evaluation script
├── interactive_reward_viewer.py
├── interactive_reward_viewer_gui.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Akame9/Bidexterous-Manipulation-of-Rubiks-Cube.git
cd Bidexterous-Manipulation-of-Rubiks-Cube
```

### 2. Create a Python environment

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

Or using Conda:

```bash
conda create -n bidex-rubiks python=3.10
conda activate bidex-rubiks
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install mujoco
```

### 4. Verify MuJoCo installation

```bash
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
```

---

## Usage

### Train PPO

```bash
python train_bidexhands_ppo.py --num_episodes 1000 --enable_viewer
```

### Train without visualization

```bash
python train_bidexhands_ppo.py --num_episodes 1000
```

### Train with custom PPO parameters

```bash
python train_bidexhands_ppo.py \
    --num_episodes 2000 \
    --lr 1e-4 \
    --gamma 0.995 \
    --eps_clip 0.1 \
    --k_epochs 15 \
    --entropy_coef 0.02 \
    --save_interval 50 \
    --model_path saved_models/my_ppo_model \
    --enable_viewer
```

### Resume from checkpoint

```bash
python train_bidexhands_ppo.py \
    --load_model saved_models/ppo_bidexhands_episode_500.pth \
    --num_episodes 1000
```

### Evaluate a trained model

```bash
python evaluate_ppo.py --model_path saved_models/ppo_bidexhands_episode_500.pth
```

---

## Configuration

Common training and environment parameters include:

| Argument | Description |
|---|---|
| `--xml_path` | Path to MuJoCo XML environment file |
| `--num_episodes` | Number of training episodes |
| `--max_episode_steps` | Maximum number of steps per episode |
| `--lr` | PPO learning rate |
| `--gamma` | Discount factor |
| `--eps_clip` | PPO clipping coefficient |
| `--k_epochs` | Number of PPO update epochs |
| `--entropy_coef` | Entropy regularization coefficient |
| `--value_coef` | Value loss coefficient |
| `--save_interval` | Checkpoint save frequency |
| `--model_path` | Output path for model checkpoints |
| `--load_model` | Path to pretrained checkpoint |
| `--enable_viewer` | Enable MuJoCo viewer |

---

## Results and Visualizations

Training logs, reward curves, and evaluation artifacts are stored in:

```text
logs/
plots/
evaluation_results_*.txt
saved_models/
```

The experiments compare:

- PPO on hinged cube tasks
- MAPPO-style training on hinged cube tasks
- PPO on freejoint cube tasks
- MAPPO-style training on freejoint cube tasks
- Generalization to unseen or longer rotation sequences

---

## Limitations

Current limitations include:

- Reward sensitivity across task variants
- Difficulty of long-horizon sequential rotations
- Instability under freejoint cube dynamics and gravity
- Potential overfitting to specific rotation sequences
- MAPPO coordination challenges when each hand learns independently
- Sim-to-real gap due to contact modeling, friction, and actuator assumptions

---

## Future Work

Potential extensions include:

- Hierarchical or skill-based RL policies
- Better reward shaping for long-horizon manipulation
- Curriculum learning from grasping to single-face and multi-face rotations
- Equivariant or symmetry-aware policy architectures
- Improved contact modeling and physical realism
- Domain randomization and sim-to-real transfer
- Centralized critic with decentralized execution for multi-agent control
- Imitation learning or demonstration-guided initialization

---

## References

- Akkaya et al., **Solving Rubik’s Cube with a Robot Hand**, arXiv:1910.07113, 2019.
- Chen et al., **Towards Human-Level Bimanual Dexterous Manipulation with Reinforcement Learning (Bi-DexHands)**, NeurIPS Benchmark Track, 2023.
- Schulman et al., **Proximal Policy Optimization Algorithms**, arXiv:1707.06347, 2017.
- Todorov et al., **MuJoCo: A Physics Engine for Model-Based Control**, 2012.

---

## Acknowledgments

This project uses MuJoCo physics simulation, Shadow Hand models, and reinforcement learning methods for dexterous robotic manipulation.
