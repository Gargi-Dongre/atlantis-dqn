# Deep Q-Learning Agent for Atari Atlantis

This project implements a Deep Q-Network (DQN) to train an AI agent that learns to play the Atari game **Atlantis (ALE/Gymnasium)** using **PyTorch**.  
The goal was to build, train, and evaluate a reinforcement learning agent that learns directly from visual frames using rewards and environment feedback.

---

## Overview

The project applies Deep Q-Learning — a value-based reinforcement learning algorithm where a neural network estimates the Q-values for each state-action pair.  
The environment was built using **Gymnasium ALE**, and training was performed on **Google Colab** with **PyTorch**.

Training was completed in two stages:

- **Phase 1 (Baseline Training - 1M Steps):**  
  Initial training to establish a working baseline using a learning rate of 1e-4, gamma = 0.99, and epsilon decay from 1.0 to 0.01 across 800k steps.

- **Phase 2 (Fine-Tuning - 3M Steps):**  
  Continued training from the saved 1M checkpoint up to 3M steps, using a lower learning rate (5e-5) and a slower epsilon decay (2M steps) for more stable convergence.

---

## Environment Preprocessing

The **Atlantis-v5** environment produces raw RGB frames of size (210×160×3).  
To make learning more efficient, the following preprocessing techniques were applied:

1. **Frame Skipping:** Every 4th frame was processed to reduce redundancy and speed up learning.  
2. **Grayscale Conversion:** Frames converted to 84×84 grayscale images to lower input complexity.  
3. **Reward Clipping:** Rewards normalized to -1, 0, or +1 to stabilize learning.  
4. **Frame Stacking:** 4 consecutive frames stacked to provide motion context.  
5. **Replay Buffer:** Stored past experiences to break correlation between consecutive steps.

These steps ensured efficient learning and stable Q-value estimation.

---

## Model Architecture

The DQN network processes four stacked 84×84 frames and outputs Q-values for each action.

| Layer | Type | Parameters | Activation |
|-------|------|-------------|-------------|
| 1 | Conv2D | 32 filters (8×8, stride 4) | ReLU |
| 2 | Conv2D | 64 filters (4×4, stride 2) | ReLU |
| 3 | Conv2D | 64 filters (3×3, stride 1) | ReLU |
| 4 | Fully Connected | 512 units | ReLU |
| 5 | Output Layer | n_actions | Linear |

---

## Training Setup

| Parameter | Value |
|------------|--------|
| Learning Rate | 1e-4 (baseline), 5e-5 (fine-tuning) |
| Gamma | 0.99 |
| Batch Size | 32 |
| Replay Buffer Size | 100,000 |
| Target Network Update | Every 5,000 steps |
| Epsilon Decay | From 1.0 → 0.01 (800K → 2M steps) |
| Total Steps | 3,000,000 |

---

## Training Process

- The agent interacts with the environment and selects actions using an **ε-greedy policy**.  
- The experience `(state, action, reward, next_state, done)` is stored in a replay buffer.  
- The model samples random mini-batches from the buffer to update Q-values.  
- The **Target Network** provides stable targets for Q-learning updates using the Bellman equation:


- The **Online Network** is optimized using the Mean Squared Error (MSE) between predicted and target Q-values.  
- Periodic evaluations (every 50K steps) measured progress using the agent’s average return.

---

## Exploration Policy

The **ε-greedy policy** was used for exploration:

- Starts with ε = 1.0 (fully random actions)  
- Linearly decays to ε = 0.01 over training  
- Encourages exploration early and exploitation later  

A **softmax policy** was also tested for comparison, but ε-greedy produced more stable and consistent improvements during training.

---

## Results

- **Baseline Training (1M steps):** Average return ≈ 1200–1300  
- **Fine-Tuned Model (3M steps):** Higher stability and smoother gameplay  
- **Average Episode Length:** ~6,500 frames  
- **Final Evaluation (ε = 0.01):** Consistent and controlled gameplay  

Training progress was tracked using **TensorBoard**, including:
- Temporal Difference (TD) loss  
- Episode rewards  
- Epsilon decay  
- Evaluation average returns  

---


