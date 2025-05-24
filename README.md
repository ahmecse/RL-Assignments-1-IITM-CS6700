# CS6700 Reinforcement Learning PA1: SARSA & Q-Learning in Grid World

This repository contains the implementation for Programming Assignment 1 of the CS6700: Reinforcement Learning course (Jan-May 2024). The assignment focuses on implementing, comparing, and analyzing two fundamental Temporal Difference (TD) learning algorithms, SARSA and Q-Learning, within a custom Grid World environment.

## Environment Description

The agent operates within a 10x10 Grid World designed for this assignment. Key characteristics include:

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
        *   **Epsilon-Greedy (ϵ-greedy):** Balances exploration and exploitation. With probability `epsilon`, a random action is chosen (explore). With probability `1 - epsilon`, the action with the highest current Q-value is chosen (exploit). Epsilon typically decays over time.
        *   **Softmax:** Selects actions based on their Q-values using a probability distribution (e.g., Boltzmann distribution). Actions with higher Q-values have a higher probability of being selected. A temperature parameter `tau` controls the randomness: high `tau` leads to more random exploration, low `tau` leads to more greedy exploitation.
    *   **Learning Rate (α):** Controls the step size for Q-value updates.
    *   **Discount Factor (γ):** Determines the importance of future rewards.
4.  **Analysis and Plotting:** For each experiment (using best hyperparameters, averaged over 5 runs, min 5000 episodes):
    *   Plot reward curves (mean and standard deviation).
    *   Plot steps to goal per episode (mean and standard deviation).
    *   Generate a heatmap of state visit counts (mean).
    *   Generate a heatmap of final Q-values and visualize the optimal policy (mean).
5.  **Reporting:** Provide a detailed comparison of the policies learned by SARSA and Q-Learning for each experiment, including justifications for hyperparameter choices and the observed behaviors.

## How to Run

*(Detailed instructions specific to this project should be added here. This typically includes:)*

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Set up environment (if necessary):**
    ```bash
    # e.g., using pip
    pip install -r requirements.txt
    ```
3.  **Run experiments:**
    ```bash
    # Example command structure (modify as needed)
    python experiments/run_experiment.py --algorithm sarsa --start_state 0,4 --p 1.0 --wind false
    python experiments/run_experiment.py --algorithm q_learning --start_state 3,6 --p 0.7 --wind false
    # ... run all 12 configurations
    ```
4.  **Generate plots and report:**
    ```bash
    # Example command (modify as needed)
    python src/utils.py --generate_plots --generate_report
    ```

## Results

*(This section should summarize the key findings from the experiments and analysis. Include links to the final report PDF and potentially showcase some key plots or heatmaps comparing SARSA and Q-Learning performance across different environment variants.)*

Example: "The final report comparing SARSA and Q-Learning performance can be found in `results/report/CS6700_PA1_Report.pdf`. Q-Learning generally converged faster in deterministic settings, while SARSA exhibited safer exploration patterns in stochastic environments..."

## Suggested Repository Structure

```
.
├── README.md           # This file.
├── requirements.txt    # Project dependencies (e.g., numpy, matplotlib).
├── src/                # Source code directory.
│   ├── algorithms/       # Implementation of SARSA and Q-Learning.
│   │   ├── sarsa.py
│   │   └── q_learning.py
│   ├── environment.py    # Grid world environment class (as provided or implemented).
│   └── utils.py          # Utility functions (plotting, hyperparameter storage, etc.).
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

This structure separates core algorithm implementations (`src/`), experiment execution logic (`experiments/`), and generated outputs (`results/`), promoting modularity and clarity.

## Dependencies

*   Python 3.x
*   NumPy
*   Matplotlib
*   (Add any other specific libraries used, e.g., `seaborn` for heatmaps)

## Contributing

Contributions to improve the implementation, add analysis, or enhance documentation are welcome. Please follow standard fork-and-pull-request workflows.

## License

This project is likely subject to the academic policies of the CS6700 course. If intended for public distribution beyond the course requirements, consider adding an open-source license like MIT.

```
# Placeholder for MIT License text if applicable
MIT License

Copyright (c) [Year] [Your Name/Team Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

