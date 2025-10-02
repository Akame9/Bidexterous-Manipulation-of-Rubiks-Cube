# Bidexhands PPO Training

This project implements Proximal Policy Optimization (PPO) for training agents to manipulate a Rubik's cube using dual Shadow Hands in MuJoCo simulation.

## Project Structure

```
├── primitive_controller/
│   └── ppo.py                 # PPO algorithm implementation
├── environment/
│   └── rubiks_cube.py         # Rubik's cube manipulation environment
├── train_bidexhands_ppo.py    # Main training script
├── requirements.txt           # Python dependencies
├── xmls/                     # MuJoCo XML model files
│   ├── bidexhands.xml        # Main environment model
│   ├── left_hand.xml         # Left shadow hand model
│   ├── right_hand.xml        # Right shadow hand model
│   ├── cube_rad.xml          # Rubik's cube model
│   └── README.md             # XML documentation
└── assets/                   # 3D models and textures
```

## Features

### PPO Agent (`primitive_controller/ppo.py`)
- **Actor-Critic Architecture**: Neural network with shared layers and separate actor/critic heads
- **PPO Algorithm**: Implements proximal policy optimization with clipping
- **Generalized Advantage Estimation (GAE)**: For stable advantage estimation
- **Experience Replay**: Memory buffer for storing and sampling experiences
- **Model Saving/Loading**: Persistent model storage and checkpointing

### Environment (`environment/rubiks_cube.py`)
- **MDP Implementation**: Complete Markov Decision Process for cube manipulation
- **State Space**: Joint positions, velocities, cube state, contact forces
- **Action Space**: Actuator controls for both hands and cube
- **Reward Function**: Multi-component reward based on manipulation success, stability, and efficiency
- **MuJoCo Integration**: Direct interface with MuJoCo physics engine

### Training Script (`train_bidexhands_ppo.py`)
- **Command-line Interface**: Configurable training parameters
- **Progress Monitoring**: Real-time training statistics and logging
- **Model Checkpointing**: Regular model saves during training
- **Evaluation**: Post-training agent evaluation
- **Visualization**: Training progress plots

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install MuJoCo** (if not already installed):
   ```bash
   pip install mujoco
   ```

3. **Verify installation**:
   ```bash
   python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
   ```

## Usage

### Basic Training

```bash
python train_bidexhands_ppo.py --num_episodes 1000 --enable_viewer
```

### Advanced Training with Custom Parameters

```bash
python train_bidexhands_ppo.py \
    --num_episodes 2000 \
    --lr 1e-4 \
    --gamma 0.995 \
    --eps_clip 0.1 \
    --k_epochs 15 \
    --entropy_coef 0.02 \
    --save_interval 50 \
    --model_path models/my_ppo_model \
    --enable_viewer
```

### Resume Training from Checkpoint

```bash
python train_bidexhands_ppo.py \
    --load_model models/ppo_bidexhands_episode_500.pth \
    --num_episodes 1000
```

### Headless Training (No Visualization)

```bash
python train_bidexhands_ppo.py --num_episodes 1000
```

## Configuration

### Environment Parameters
- `--xml_path`: Path to MuJoCo XML model file
- `--max_episode_steps`: Maximum steps per episode
- `--enable_viewer`: Enable visual rendering

### Training Parameters
- `--num_episodes`: Number of training episodes
- `--lr`: Learning rate (default: 3e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--eps_clip`: PPO clipping parameter (default: 0.2)
- `--k_epochs`: Policy update epochs (default: 10)
- `--entropy_coef`: Entropy coefficient (default: 0.01)
- `--value_coef`: Value function loss coefficient (default: 0.5)

### Model Management
- `--save_interval`: Model save frequency
- `--model_path`: Model save path
- `--load_model`: Path to pre-trained model

## Environment Details

### State Space
The environment state includes:
- **Joint Positions**: All joint angles (qpos)
- **Joint Velocities**: All joint velocities (qvel)
- **Cube State**: Position (3D), orientation (quaternion), velocities
- **Contact Forces**: Hand-cube interaction forces
- **Previous Actions**: Last executed actions

### Action Space
Actions control:
- **Left Hand Actuators**: 16 actuators for left Shadow Hand
- **Right Hand Actuators**: 16 actuators for right Shadow Hand
- **Cube Actuators**: 6 actuators for cube manipulation

### Reward Function
The reward function combines:
- **Cube Manipulation** (40%): Based on cube movement and stability
- **Grasping Quality** (30%): Based on contact forces
- **Action Efficiency** (20%): Penalizes excessive actions
- **System Stability** (10%): Penalizes unstable configurations

## Training Tips

1. **Start with Viewer**: Use `--enable_viewer` to monitor training visually
2. **Adjust Learning Rate**: Lower learning rates (1e-4) for more stable training
3. **Monitor Rewards**: Watch for consistent reward improvement
4. **Save Regularly**: Use `--save_interval` to prevent loss of progress
5. **Evaluate Periodically**: Test trained models on evaluation episodes

## Troubleshooting

### Common Issues

1. **MuJoCo Installation**: Ensure MuJoCo is properly installed and accessible
2. **XML Path**: Verify the path to `xmls/bidexhands.xml` is correct
3. **Memory Issues**: Reduce batch size or use CPU if GPU memory is insufficient
4. **Viewer Issues**: Disable viewer if running on headless systems

### Performance Optimization

1. **Device Selection**: Use GPU (`--device cuda`) for faster training
2. **Batch Size**: Adjust based on available memory
3. **Episode Length**: Shorter episodes for faster iteration
4. **Parallel Training**: Consider multiple workers for data collection

## Results and Evaluation

The training script provides:
- **Real-time Statistics**: Episode rewards, lengths, and losses
- **Training Plots**: Visual progress tracking
- **Model Checkpoints**: Regular saves during training
- **Evaluation Results**: Post-training performance assessment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

## Acknowledgments

- MuJoCo physics engine
- Shadow Hand models
- PPO algorithm implementation
- Rubik's cube manipulation research

