# CS6700 Reinforcement Learning PA1: SARSA & Q-Learning

This repository contains the implementation for Programming Assignment 1 of the CS6700: Reinforcement Learning course (Jan-May 2024).
The primary goal of this assignment is to implement and compare two fundamental Temporal Difference (TD) learning algorithms: SARSA and Q-Learning, applied to variants of the Grid World problem.

## Environment Description

The agent operates within a 10x10 Grid World. Key characteristics include:

*   **Actions:** The agent can choose one of four deterministic actions: 'up', 'down', 'left', 'right'.
*   **Transitions:** Action outcomes can be stochastic. The agent moves in the intended direction with probability `p`. With probability `(1-p)*b`, it moves West relative to the intended direction, and with probability `(1-p)*(1-b)`, it moves East.
*   **Wind:** An optional wind effect can push the agent one cell to the right with a probability of 0.4 after its initial move.
*   **States:** The grid contains several types of states:
    *   **Start State:** The agent's initial position (varies per experiment).
    *   **Goal States:** Target states (3 total). Reaching a goal yields a reward of +10.
    *   **Obstructed States:** Walls that block movement.
    *   **Bad States:** Entering incurs a penalty of -6.
    *   **Restart States:** Entering incurs a high penalty of -100 and teleports the agent back to the start state.
    *   **Normal States:** All other states, incurring a penalty of -1 upon entry.
*   **Episode Termination:** An episode ends when the agent reaches a goal state or exceeds 100 timesteps.
*   **Boundary Conditions:** Attempting to move off the grid results in no change in state.

## Algorithms Implemented

1.  **SARSA (State-Action-Reward-State-Action):** An on-policy TD control algorithm.
2.  **Q-Learning:** An off-policy TD control algorithm.

## Tasks and Experiments

The assignment involves the following core tasks:

1.  **Implementation:** Implement both SARSA and Q-Learning algorithms.
2.  **Experimentation:** Run a total of 12 experiments, covering:
    *   **Algorithms:** SARSA and Q-Learning.
    *   **Start States:** (0, 4) and (3, 6).
    *   **Environment Variants:**
        *   Deterministic steps (`p=1.0`), no wind.
        *   Stochastic steps (`p=0.7`), no wind.
        *   Deterministic steps (`p=1.0`), with wind.
3.  **Hyperparameter Tuning:** For each experiment, determine the optimal hyperparameters:
    *   Action selection policy: Epsilon-greedy (ϵ) or Softmax (τ).
    *   Learning rate (α).
    *   Discount factor (γ).
4.  **Analysis and Plotting:** For each experiment (using best hyperparameters, averaged over 5 runs, min 5000 episodes):
    *   Plot reward curves (mean and standard deviation).
    *   Plot steps to goal per episode (mean and standard deviation).
    *   Generate a heatmap of state visit counts (mean).
    *   Generate a heatmap of final Q-values and visualize the optimal policy (mean).
5.  **Reporting:** Provide a detailed comparison of the policies learned by SARSA and Q-Learning for each experiment, including justifications for hyperparameter choices and the observed behaviors.

## How to Run

*(Instructions on setting up the environment and running the code would go here. This typically includes dependencies, environment setup commands, and script execution details.)*

## Results

*(This section would typically summarize the key findings, link to the final report (PDF), and potentially showcase some of the generated plots or heatmaps.)*



## Suggested Repository Structure

```
.
├── README.md           # The top-level README for reviewers and developers.
├── requirements.txt    # Project dependencies (e.g., numpy, matplotlib).
├── src/                # Source code directory.
│   ├── algorithms/       # Implementation of SARSA and Q-Learning.
│   │   ├── sarsa.py
│   │   └── q_learning.py
│   ├── environment.py    # Grid world environment class (or link to provided code).
│   └── utils.py          # Utility functions (e.g., plotting, hyperparameter storage).
├── experiments/        # Scripts to run the experiments.
│   ├── run_experiment.py # Main script to run a specific experiment configuration.
│   └── configs/          # Configuration files for different experiments (optional).
├── results/            # Directory to store results.
│   ├── plots/            # Generated plots (reward curves, heatmaps).
│   │   ├── sarsa/
│   │   └── q_learning/
│   ├── data/             # Raw data from experiments (optional).
│   └── report/           # Final report.
│       └── CS6700_PA1_Report.pdf
└── notebooks/          # Jupyter notebooks for exploration or visualization (optional).
    └── analysis.ipynb
```

This structure separates the core algorithm implementations (`src/`), experiment execution logic (`experiments/`), and generated outputs (`results/`), promoting modularity and clarity.

