# WorldModel

This project utilizes the Gymnasium FrozenLake environment and implements a Q-learning algorithm to train on maps of various sizes. In the process, it collects and processes trajectories (paths) according to specific rules:

Unique Path Saving: Only unique paths (based on the sequence of actions) are saved.
Handling Hole (Failure) Paths:
50% probability: Save the current path immediately.
50% probability: Append two random extra steps to the path before saving.
Handling Successful or Other Failed Paths: The path is saved directly.
Additionally, each step’s state (as a screenshot) is saved as an image, and the complete path information is recorded in a CSV file for later analysis or replay.

