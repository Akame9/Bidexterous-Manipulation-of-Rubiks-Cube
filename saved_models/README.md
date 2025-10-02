# Saved Models Directory

This directory contains trained PPO models for the Rubik's Cube manipulation task.

## Model Files

- `ppo_bidexhands_final.pth` - Final trained model after all episodes
- `ppo_bidexhands_episode_X.pth` - Checkpoint models saved during training
- `ppo_bidexhands_interrupted.pth` - Model saved if training was interrupted

## Usage

To evaluate a trained model:

```bash
# Visual evaluation
python utils/evaluate_trained_model.py --model_path saved_models/ppo_bidexhands_final.pth --mode viewer

# Detailed analysis
python utils/evaluate_trained_model.py --model_path saved_models/ppo_bidexhands_final.pth --mode analyze
```

## Model Information

Each model file contains:
- Policy network weights (actor)
- Value network weights (critic)
- Optimizer state
- Training statistics
