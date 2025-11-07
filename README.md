# Deep Q-Learning on Atari Atlantis (Gymnasium ALE)

This repository contains my implementation of a Deep Q-Network (DQN) agent trained on the **Atlantis-v5** Atari environment using **Gymnasium ALE** and **PyTorch**.

## Project Overview
I implemented Deep Q-Learning in two phases:
- **Phase 1:** Baseline training up to 1M steps  
- **Phase 2:** Continued fine-tuning to 3M steps with lower learning rate and slower epsilon decay for stable performance.

## Key Techniques
- Frame skipping to speed up learning  
- Grayscale 84×84 resizing to reduce input complexity  
- Reward clipping (−1, 0, 1) for stability  
- Frame stacking of 4 frames to provide motion context  
- Epsilon-greedy exploration for balance between exploration and exploitation  

## Results
- Average evaluation return: ~1300 at 3M steps  
- Training progress monitored via TensorBoard  
- Final checkpoint and gameplay video recorded for demonstration  

## Files
- `atlantis_dqn.ipynb` – Main training and resume code  
- `requirements.txt` – Dependencies  
- `Deep_Q_Learning_Atari_Assignment_GargiDongre.pdf` – Rubric and report  
- `atlantis_dqn_gameplay.mp4` – Gameplay demonstration  
- `atlantis_explainer.mp4` – Project explanation video  

## Requirements
