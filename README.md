# Hyper Parameter Optimization: Hyper Gradient Descent Through Checkpointing

<span style="font-size: 18px;">In this repository I reproduce some of the results presented in [Gradient-based Hyperparameter Optimization through Reversible Learning](https://arxiv.org/abs/1502.03492) but instead of performing the backpropagation of the training loop through exact representation, I used the checkpointing method proposed in [Revolve paper](https://dl.acm.org/doi/pdf/10.1145/347837.347846) to restore parameter values at a given iteration.

<div style="display: flex; justify-content: space-around;">
  <img src="results_learning_rate/meta_learning_cp.png" alt="Training Loss Curve" width="300" style="margin-right: 20px;">
  <img src="results_learning_rate/initialvsfinal_cp.png" alt="Gradient Descent Visualization" width="300">
  <img src="results_learning_rate/learning_schedule_cp.png" alt=" learning schedule" width="300">
</div>

<br><br>