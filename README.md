# RL Racing Game 🚗💨

A beautiful, modern, and fully-featured 2D top-down racing game built in Python, powered by Reinforcement Learning (RL). The agent learns to drive around a complex track, cross checkpoints in order, and finish laps as quickly as possible—all using Deep Q-Learning!

---

## Table of Contents
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Reinforcement Learning Details](#reinforcement-learning-details)
- [Installation & Setup](#installation--setup)
- [How to Train the Agent](#how-to-train-the-agent)
- [How to Play Manually](#how-to-play-manually)
- [Saving & Resuming Training](#saving--resuming-training)
- [Customization](#customization)
- [Credits](#credits)

---

## Features
- **Physics-based 2D racing** with realistic car handling, drifting, and wall collisions
- **Procedurally generated checkpoints** and complex track
- **Deep Q-Network (DQN) RL agent** that learns to race from scratch
- **Reward shaping** for progress, checkpoint crossing, and lap completion
- **Automatic model saving and resuming**
- **Beautiful Pygame visualization** of the agent's learning
- **Easy to customize** for new tracks, reward functions, or RL algorithms

---

## Project Architecture

```
rlcars/
├── racing_game.py      # Main game and RL environment
├── rl_agent.py         # DQN agent, replay memory, and training logic
├── requirements.txt    # All dependencies
├── README.md           # This file
└── ...                 # (Other assets, models, etc.)
```

- **racing_game.py**: Contains the game logic, environment, reward system, and training loop. Implements the OpenAI Gym-like `Game` class for RL.
- **rl_agent.py**: Contains the DQN neural network, experience replay buffer, and agent logic.

---

## Reinforcement Learning Details

### State Representation
- **Lidar sensors**: 8 rays cast from the car to detect distance to walls
- **Car dynamics**: Speed and angular velocity
- **Goal info**: Distance and angle to the next checkpoint (in order)

### Action Space
- 9 discrete actions:
  - No-op
  - Accelerate
  - Brake
  - Steer left/right
  - Accelerate + steer left/right
  - Brake + steer left/right

### Reward Function
- **+100** for crossing a checkpoint (in order)
- **+100** for completing a lap (after all checkpoints)
- **Progress reward**: Proportional to how much closer the car gets to the next checkpoint each step
- **-0.1** per step (time penalty)
- **-20** for crashing into a wall

### Model Saving & Resuming
- The agent's model and exploration state are saved every 100 episodes
- On startup, the latest checkpoint is loaded automatically (if available)

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone git@github.com:nikusrotuslomatus/rlracing.git
   cd rlracing
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Set up your SSH keys for GitHub** if you want to push changes.

---

## How to Train the Agent

Just run:
```bash
python racing_game.py
```
- The agent will start learning from scratch or resume from the latest checkpoint.
- Every 20th episode, you'll see a live visualization of the agent's driving.
- Models are saved every 100 episodes for easy resuming.

---

## How to Play Manually

If you want to drive the car yourself, you can modify the `racing_game.py` main block to call the `main()` function instead of `train_rl_agent()`. Use the arrow keys to control the car:
- **Up Arrow**: Accelerate
- **Down Arrow**: Brake/Reverse
- **Left/Right Arrows**: Steer

---

## Saving & Resuming Training
- The agent's model and exploration state are saved as `dqn_model_epXXX.pth` every 100 episodes.
- On startup, the latest checkpoint is loaded automatically.
- To continue training, just run the script again—no manual intervention needed!

---

## Customization
- **Track & Checkpoints**: Edit the `OUTER_TRACK` and `INNER_TRACK` polygons and the number of checkpoints in `racing_game.py`.
- **Reward Function**: Tweak the reward logic in the `step()` method for different behaviors.
- **RL Algorithm**: Swap out the DQN in `rl_agent.py` for other algorithms (e.g., PPO, A2C) if desired.
- **Visualization**: Enhance the `render()` method for more overlays, stats, or effects.

---

## Credits
- Built with [Pygame](https://www.pygame.org/), [Pymunk](http://www.pymunk.org/), and [PyTorch](https://pytorch.org/)
- RL architecture inspired by OpenAI Gym and classic DQN papers
- Created by [nikusrotuslomatus](https://github.com/nikusrotuslomatus/)

---

Enjoy racing and RL research! 🚗🏁 
