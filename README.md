# CS6700 Reinforcement Learning PA1: SARSA & Q-Learning in Grid World
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](requirements.txt)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-brightgreen.svg)](https://pypi.org/project/gymnasium/)
[![NumPy](https://img.shields.io/badge/numpy-1.24+-orange.svg)](https://pypi.org/project/numpy/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.5+-blueviolet.svg)](https://pypi.org/project/matplotlib/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-yellow.svg)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.13+-red.svg)](https://pytorch.org/)
[![tqdm](https://img.shields.io/badge/tqdm-progress--bar-lightgrey.svg)](https://pypi.org/project/tqdm/)

This repository contains the implementation for Programming Assignment 1 of the CS6700: Reinforcement Learning course (Jan-May 2024). The assignment focuses on implementing, comparing, and analyzing two fundamental Temporal Difference (TD) learning algorithms, SARSA and Q-Learning, within a custom Grid World environment.

![Report Organization](Images/The%20report%20can%20be%20organized.png)

## Environment Description

The agent operates within a 10x10 Grid World designed for this assignment. Key characteristics include:

![Grid Example](Images/An%20example%20grid%20with%20start%20point%20at%20(0,4).png)

*   **Actions:** The agent can choose one of four deterministic actions: `up`, `down`, `left`, `right`.
*   **Transitions:** Action outcomes can be stochastic based on parameters `p` and `b`. The agent moves in the intended direction (North) with probability `p`. With probability `(1-p)*b`, it moves West relative to the intended direction, and with probability `(1-p)*(1-b)`, it moves East.
*   **Wind:** An optional wind effect can push the agent one cell to the right with a probability of 0.4 after its initial move.
*   **States:** The grid contains several types of states:
    *   **Start State:** The agent's initial position (fixed per experiment, e.g., (0, 4) or (3, 6)).
    *   **Goal States:** Target states (3 total). Reaching a goal yields a reward of +10.
    *   **Obstructed States:** Walls that block movement; transitions into these result in no state change.
    *   **Bad States:** Entering incurs a penalty of -6.
    *   **Restart States:** Entering incurs a high penalty of -100 and teleports the agent back to the start state without ending the episode.
    *   **Normal States:** All other states, incurring a penalty of -1 upon entry.
*   **Episode Termination:** An episode ends when the agent reaches a goal state or exceeds 100 timesteps.
*   **Boundary Conditions:** Attempting to move off the grid results in no change in state.
*   **Rewards:** +10 (Goal), -1 (Normal), -6 (Bad), -100 (Restart).

![Action Directions](Images/The%20intended%20direction%20of%20the%20action%20chosen%20is%20considered%20as%20North.png)

## Algorithms Implemented

This assignment requires the implementation of two TD learning algorithms:

1.  **SARSA (State-Action-Reward-State-Action):** An on-policy TD control algorithm. It learns the action-value function based on the action actually taken by the current policy (often epsilon-greedy or softmax) in the next state.
2.  **Q-Learning:** An off-policy TD control algorithm. It learns the action-value function based on the estimated optimal action in the next state, regardless of the action actually taken. Q-Learning aims to directly approximate the optimal action-value function.

### Q-Learning Update Rule

The core of Q-Learning involves updating the Q-value for a state-action pair using the Bellman equation:

```
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a'(Q(s', a')) - Q(s, a)]
```

Where:
*   `Q(s, a)`: The estimated value of taking action `a` in state `s`.
*   `alpha`: The learning rate, determining how much new information overrides old information.
*   `r`: The immediate reward received after taking action `a` in state `s`.
*   `gamma`: The discount factor, valuing future rewards.
*   `s'`: The next state observed after taking action `a`.
*   `max_a'(Q(s', a'))`: The maximum estimated Q-value for the next state `s'` over all possible actions `a'` (the greedy choice).

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
3.  **Hyperparameter Tuning & Action Selection:** For each experiment, determine the optimal hyperparameters and action selection policy:
    *   **Action Selection Policies:**
        *   **Epsilon-Greedy (ϵ-greedy):** Balances exploration and exploitation.
        *   **Softmax:** Selects actions based on their Q-values using a probability distribution.
    *   **Learning Rate (α):** Controls the step size for Q-value updates.
    *   **Discount Factor (γ):** Determines the importance of future rewards.
4.  **Analysis and Plotting:** For each experiment (using best hyperparameters, averaged over 5 runs, min 5000 episodes):
    *   Plot reward curves (mean and standard deviation).
    *   Plot steps to goal per episode (mean and standard deviation).
    *   Generate a heatmap of state visit counts (mean).
    *   Generate a heatmap of final Q-values and visualize the optimal policy (mean).
5.  **Reporting:** Provide a detailed comparison of the policies learned by SARSA and Q-Learning for each experiment, including justifications for hyperparameter choices and the observed behaviors.

## How to Run

1.  **Set up environment:** Install necessary dependencies listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
2.  **Explore the Code:** The core environment logic is in `Notebooks/gridworld_pa1.py`. The experiments, algorithm implementations (SARSA, Q-Learning), plotting functions, and hyperparameter tuning logic are contained within the Jupyter notebook `Notebooks/notebook.ipynb`.
3.  **Run Experiments:** Open and run the cells within `Notebooks/notebook.ipynb` using a Jupyter environment (like Jupyter Lab or VS Code with Python extensions). The notebook is structured to run each of the 12 experiments, perform hyperparameter tuning (if enabled), train the models, and generate/save the result plots.

## Results

The results from running the 12 core experiments (averaged over 5 runs for 5000 episodes each) are pre-generated and stored as PDF files within the `Results/5K-episodes/` directory. The full analysis and discussion can be found in the report `CS6700_PA1_Report.pdf` and the execution flow within `Notebooks/notebook.ipynb`.

### Result Directory Structure (`Results/5K-episodes/`)

The subdirectories are named using a four-part code: `X.Y.Z.W`, corresponding to the experiment parameters:

*   **X - Environment Variant:**
    *   `1`: wind=False (clear), p=1.0 (deterministic step)
    *   `2`: wind=False (clear), p=0.7 (stochastic step)
    *   `3`: wind=True (windy), p=1.0 (deterministic step)
*   **Y - Start State:**
    *   `1`: Start State (0, 4)
    *   `2`: Start State (3, 6)
*   **Z - Algorithm:**
    *   `1`: SARSA
    *   `2`: Q-Learning
*   **W - Action Selection Policy:**
    *   `1`: Softmax
    *   `2`: Epsilon-Greedy

**Example:** `Results/5K-episodes/3.1.2.1/` contains results for: Windy environment (X=3), Start State (0, 4) (Y=1), Q-Learning algorithm (Z=2), and Softmax policy (W=1).

### Result File Types (within each `X.Y.Z.W` directory)

*   `grid_world.pdf`: Visualization of the specific Grid World configuration.
*   `optimal_path.pdf`: Visualization of the optimal path found after training.
*   `q_values.pdf`: Heatmap of learned Q-values and optimal actions.
*   `reward.pdf`: Plot of reward per episode (mean/std dev over runs).
*   `state_visits.pdf`: Heatmap of state visit counts during training.
*   `steps.pdf`: Plot of steps taken per episode (mean/std dev over runs).

### Example Results

Here are some example plots corresponding to specific experiments:

*   **SARSA vs Q-Learning Rewards (Deterministic, No Wind, Start (0,4), Softmax):**
    *   SARSA:
        ![SARSA Rewards](https://raw.githubusercontent.com/ahmecse/RL-Assignments-1-IITM-CS6700/main/png_results/SARSA_Det_NoWind_Start04_Softmax_Rewards.png)
    *   Q-Learning:
        ![Q-Learning Rewards](https://raw.githubusercontent.com/ahmecse/RL-Assignments-1-IITM-CS6700/main/png_results/QLearning_Det_NoWind_Start04_Softmax_Rewards.png)

*   **Optimal Path (Q-Learning, ε-Greedy, Windy, Start (3,6)):**
    ![Optimal Path](https://raw.githubusercontent.com/ahmecse/RL-Assignments-1-IITM-CS6700/main/png_results/QLearning_Windy_Det_Start36_EpsilonGreedy_OptimalPath.png)

*   **State Visits (Stochastic, No Wind, Start (3,6), ε-Greedy):**
    *   SARSA:
        ![SARSA Visits](https://raw.githubusercontent.com/ahmecse/RL-Assignments-1-IITM-CS6700/main/png_results/SARSA_Stochastic_NoWind_Start36_EpsilonGreedy_StateVisits.png)
    *   Q-Learning:
        ![Q-Learning Visits](https://raw.githubusercontent.com/ahmecse/RL-Assignments-1-IITM-CS6700/main/png_results/QLearning_Stochastic_NoWind_Start36_EpsilonGreedy_StateVisits.png)

*Note: The full set of plots for all 12 experiments can be found in the `Results/5K-episodes/` directory as PDF files.*

## Repository Structure

```
.
├── CS6700_PA1_Report.pdf # The detailed assignment report.
├── Images/                 # Supporting images used in README and Report.
│   ├── An example grid with start point at (0,4).png
│   ├── The intended direction of the action chosen is considered as North.png
│   └── The report can be organized.png
├── Notebooks/              # Jupyter notebook and Python script.
│   ├── gridworld_pa1.py    # Defines the GridWorld environment class.
│   └── notebook.ipynb      # Main notebook with implementations, experiments, and plotting.
├── README.md               # This file.
├── Results/                # Contains generated plots and potentially other outputs.
│   ├── 1.1.1_egreedy/      # Example of other result folders (structure may vary)
│   ├── ...                 # Other similar folders
│   ├── 5K-episodes/        # Main results for the 12 experiments (5000 episodes).
│   │   ├── 1.1.1.1/        # Subdirs named based on experiment parameters (X.Y.Z.W)
│   │   │   ├── grid_world.pdf
│   │   │   ├── optimal_path.pdf
│   │   │   ├── q_values.pdf
│   │   │   ├── reward.pdf
│   │   │   ├── state_visits.pdf
│   │   │   └── steps.pdf
│   │   ├── ...             # Other X.Y.Z.W directories
│   │   └── 3.2.2.2/
│   └── environment/        # Base environment visualization.
│       └── grid_world.pdf
└── requirements.txt        # Project dependencies.
```

