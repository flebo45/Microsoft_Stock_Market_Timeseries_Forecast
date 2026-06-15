# Project Conclusions: Deep Learning for Time-Series Forecasting

This document summarizes the empirical findings from our comparative analysis of various Recurrent Neural Network architectures on the Microsoft stock dataset. By systematically isolating and altering network topologies and hyperparameters, we observed distinct behaviors regarding gradient flow, model capacity, and convergence stability.

---

## 1. Architectural Comparisons: Capacity vs. Degradation

In our primary experiments, we evaluated 8 distinct network topologies. The goal was to observe how depth, gating mechanisms, and residual connections affect a network's ability to learn from 1-dimensional financial time-series data.

**Expectations:**

- **Shallow Models (`Vanilla_RNN`, `Shallow_LSTM`, `Shallow_GRU`):** Expected to perform reasonably well. Time-series data with a single feature often does not require massive parameter counts.
- **Deep Models (`Deep_LSTM`, `Deep_GRU`, `Super_Deep_LSTM`):** Expected to struggle. Stacking recurrent layers creates a long mathematical path for backpropagation, risking vanishing gradients (model collapse) or severe overfitting.
- **Residual Models (`Shallow_LSTM_Res`, `Deep_LSTM_Res`):** Expected to rescue deeper architectures by providing gradient "skip-connections," allowing the network to bypass degraded layers.

**Visual Observations & Results:**

1. **The Success of Simplicity:** The `Shallow_GRU`, `Shallow_LSTM`, and `Vanilla_RNN` consistently traced the shape of the stock price best. Their limited parameter space prevented them from getting stuck in complex local minima, adhering to Occam's Razor.
2. **Model Collapse in Deep Networks:** As predicted, the `Deep_LSTM`, `Deep_GRU`, and `Super_Deep_LSTM` suffered from complete mathematical failure (Model Collapse). In the visualizations, these models flatline near the bottom of the graph. Because the sequence is only 1D, the deep spatial layers failed to extract meaningful hierarchical features, causing the gradients to vanish and the model to simply output the historical mean.
3. **The Power of Residual Connections:** The `Deep_LSTM_Res` (purple line) successfully avoided the flatline collapse that killed the standard `Deep_LSTM`. By adding the mathematical operation `Output = Layer(x) + x`, gradients were able to flow unimpeded, allowing the deeper network to track the stock's shape.

---

## 2. Hyperparameter Analysis: The Impact of Learning Rate

|                     Learning Rate = 0.001                      |                     Learning Rate = 0.003                      |                     Learning Rate = 0.010                     |
| :------------------------------------------------------------: | :------------------------------------------------------------: | :-----------------------------------------------------------: |
| ![LR 0.001](results/stock_predictions_2026-06-15_15-42-48.png) | ![LR 0.003](results/stock_predictions_2026-06-15_15-58-13.png) | ![LR 0.01](results/stock_predictions_2026-06-15_16-02-00.png) |

To understand optimizer dynamics, we trained the exact same 8 architectures using the Adam optimizer across three different learning rates: `0.001`, `0.003`, and `0.01`.

**1. Learning Rate = 0.001 (The Baseline):**
At this low, conservative learning rate, the shallow models converged stably. However, the `Shallow_LSTM_Res` (orange) and all deep networks flatlined. The optimizer took steps that were too small to escape early local minima, causing the networks with more complex loss landscapes to stall out early in training.

**2. Learning Rate = 0.003 (The Sweet Spot for ResNets):**
Increasing the learning rate to 0.003 yielded a fascinating result: the `Shallow_LSTM_Res` (orange line) suddenly "woke up." With slightly larger optimizer steps, it escaped the local minima that trapped it at 0.001 and became one of the best-performing models on the graph. The shallow models remained stable, proving their robustness. However, the deep models (green, cyan, brown) still flatlined, proving their failure is structural, not just a hyperparameter issue.

**3. Learning Rate = 0.01 (Aggressive Overshooting):**
At this high learning rate, we observed signs of divergence. While the shallow models managed to hold their shape, the flatlines of the collapsed deep models (`Super_Deep_LSTM`, `Deep_GRU`) actually dropped even lower on the Y-axis. The massive optimizer steps caused the gradients to violently bounce across the loss landscape, actively worsening the model collapse and pushing the network's baseline predictions down.

---

## 3. Experiment: Input Sequence Length (Chunk Size)

|                        Chunk Size = 10                         |                        Chunk Size = 21                         |                        Chunk Size = 50                         |
| :------------------------------------------------------------: | :------------------------------------------------------------: | :------------------------------------------------------------: |
| ![Chunk 10](results/stock_predictions_2026-06-15_16-27-13.png) | ![Chunk 21](results/stock_predictions_2026-06-15_15-42-48.png) | ![Chunk 50](results/stock_predictions_2026-06-15_16-31-28.png) |

To evaluate how horizontal temporal memory affects forecasting accuracy, we tested the architectures using three different historical lookback windows: 10 days, 21 days (baseline), and 50 days.

**Expectation:** Shorter sequences (10 days) might not provide enough context for the models to recognize broader market momentum. Longer sequences (50 days) should give gated networks (LSTM/GRU) rich contextual data to improve predictions. However, expanding the window to 50 days should actively harm the Vanilla RNN, as calculating backpropagation over 50 sequential steps almost guarantees vanishing gradients.

**Visual Observations & Results:**

1. **Chunk Size = 10 (Insufficient Context):** At a 10-day lookback, the Shallow LSTM and Shallow GRU performed notably worse than the baseline, underfitting the upward trend. Ten days of data simply does not contain enough macro-economic context to predict long-term momentum. Interestingly, the Vanilla RNN performed relatively well here, because gradients can easily survive a short 10-step backpropagation journey.
2. **Chunk Size = 21 (The Balanced Baseline):** This provided a functional middle ground where the gated models captured enough trend momentum to perform adequately without overwhelming the computational graph.
3. **Chunk Size = 50 (The Memory Test):** At a 50-day lookback, we observed the most significant divergence in architectural capability:
   - **LSTM & GRU Success:** The Shallow LSTM and Shallow GRU achieved their best visual performance, tracking much closer to the actual stock line. Their internal gating mechanisms successfully filtered and retained relevant momentum data across the long 50-day sequence.
   - **Vanilla RNN Degradation:** As theoretically expected, the Vanilla RNN degraded and lagged significantly further behind the gated models compared to the 10-day test. This is physical proof of the **Vanishing Gradient Problem**; without gates to manage memory, the network "forgot" the earliest days of the 50-day window, making its predictions erratic and reactive.
   - **Deep Model Failure:** The deep, multi-layer models (Deep LSTM, Deep GRU) remained completely flatlined across all sequence lengths. This proves that their failure is related to _vertical_ spatial depth (layer-to-layer gradient degradation), not _horizontal_ temporal length.

---

## 4. Experiment: Hidden Unit Dimensionality (Network Width)

|                        Hidden Size = 16                         |                        Hidden Size = 64                         |                        Hidden Size = 128                         |
| :-------------------------------------------------------------: | :-------------------------------------------------------------: | :--------------------------------------------------------------: |
| ![Hidden 16](results/stock_predictions_2026-06-15_16-41-05.png) | ![Hidden 64](results/stock_predictions_2026-06-15_15-42-48.png) | ![Hidden 128](results/stock_predictions_2026-06-15_16-44-15.png) |

To evaluate how representational capacity affects forecasting, we tested the architectures across three different layer widths: 16 neurons, 64 neurons (baseline), and 128 neurons.

**Expectation:** Decreasing the hidden size to 16 should result in **Underfitting**, as the network lacks the mathematical parameters to map the complex variance of the stock market. Increasing it to 128 should result in **Overfitting**, where the massive parameter count allows the network to memorize training noise, leading to highly erratic, high-variance predictions on the test set. Furthermore, widening the network should not fix the deep models, as their failure is related to depth (vanishing gradients), not width.

**Visual Observations & Results:**

1. **Hidden Size = 16 (Underfitting):** At only 16 neurons, the shallow models (`Vanilla_RNN`, `Shallow_LSTM`, `Shallow_GRU`) visibly lacked the capacity to capture the amplitude of the stock's upward momentum. Their prediction lines are relatively smooth but are compressed severely downward, demonstrating a classic high-bias (underfit) scenario.
2. **Hidden Size = 64 (The Baseline):** This configuration provided the optimal mathematical balance, allowing the shallow models to capture the general trend without becoming overly erratic.
3. **Hidden Size = 128 (High Variance / Overfitting):** At 128 neurons, the network's capacity drastically increased.
   - **Shallow LSTM & GRU:** These models pushed much higher on the Y-axis, getting closer to the absolute magnitude of the true stock price. However, the prediction lines became noticeably sharper and more erratic. The networks used their massive parameter counts to fit the noisy micro-fluctuations of the training data, leading to a loss of smooth generalization on the test set.
   - **Deep Model Persistent Failure:** Even with 128 neurons per layer, the deep architectures (`Deep_LSTM`, `Super_Deep_LSTM`, `Deep_GRU`) still completely flatlined at the bottom of the graph. This empirically proves that their failure is strictly a vertical gradient degradation issue (depth), and throwing more parameters (width) at the problem does absolutely nothing to fix it.

---

## 5. Final Experiment: The Synthesized "Optimal" Combination

![FINAL](results/stock_predictions_2026-06-15_16-50-17.png)

To conclude our hyperparameter tuning, we tested a combined configuration designed to maximize reactivity and capacity: **Chunk Size = 12, Learning Rate = 0.003, and Hidden Units = 128.**

**Expectation:** \* A short 12-day window forces the network to focus on immediate, short-term price action.

- A massive 128-neuron width provides high representational capacity.
- A 0.003 learning rate allows the optimizer to take aggressive steps.
  Combined, we expect highly reactive (spiky) predictions. We also expect the Residual models to perform well (due to the 0.003 LR), but deep standard networks to fail.

**Visual Observations & Results:**

1. **The Ultimate Winner (`Shallow_LSTM_Res`):** The Shallow LSTM with Residual Connections (orange dashed line) achieved the best visual performance of the entire project. The combination of a higher learning rate (allowing it to escape local minima) and high parameter capacity (128 units) allowed it to closely track the aggressive upward momentum of the actual stock (black line).
2. **High Reactivity vs. Smoothness:** Compared to the 50-day chunk size experiments, the prediction lines here are noticeably sharper and more jagged. Because the network only looks back 12 days, it reacts instantly to short-term market noise rather than smoothing out a long-term macroeconomic trend.
3. **The Absolute Limit of Deep Networks:** Even with optimal learning rates and massive layer widths, the `Super_Deep_LSTM`, `Deep_LSTM`, and `Deep_GRU` completely collapsed (flatlining between $50 and $75). This provides our final, conclusive proof: in 1-dimensional financial time-series, stacking deep recurrent layers without skip-connections mathematically destroys the gradients. You cannot hyperparameter-tune your way out of the Vanishing Gradient Problem.
