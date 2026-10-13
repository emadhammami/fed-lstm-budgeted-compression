# fed-lstm-budgeted-compression

> **Historical exploratory prototype.** The current example script takes the first value of each dataset-level `target` series and creates sliding windows across rows rather than consecutive hourly observations. Its reported MAE values therefore should not be treated as validated hourly electricity-load forecasting results. Proper temporal reconstruction and chronological evaluation are needed before drawing performance conclusions. The code is retained for inspection.


Open-source implementation of **compressed federated learning** for time-series forecasting using an LSTM with **per-client adaptive budgeted compression** on the [Electricity Load Diagrams](https://huggingface.co/datasets/tulipa762/electricity_load_diagrams) dataset.

## How this differs from previous repos

| | [Repo 1](https://github.com/emadhammami/Compressed-Federated-Time-Series-Forecasting) | [Repo 2](https://github.com/emadhammami/Compressed-FL-Univariate-LSTM-Forecasting) | [Repo 3](https://github.com/emadhammami/Fed-Multivariate-LSTM-DeltaComp) | **This repo** |
|---|---|---|---|---|
| **Input** | Static cat + calendar | Raw load (univariate) | Load history + calendar | Static cat + calendar |
| **Seq length** | 10 | 24 | 24 | 10 |
| **Local epochs** | 2 | 10 | 2 | 2 |
| **Compression** | Fixed `kf=1e-3` sparsity | Fixed `kf=1e-3` sparsity | Fixed `kf=1e-3` sparsity | **Per-client KB budget** (binary search threshold) |
| **Client budgets** | — | — | — | `[0.3, 0.4, 0.8]` KB |
| **Key novelty** | Baseline | Univariate sequences | Multivariate sequences | **Adaptive budgeted compression per client** |

### What is "budgeted compression"?

Instead of using a fixed sparsity ratio (`kf`), each client is assigned a **maximum upload budget in KB**. The `DeltaCompressor` uses **binary search on the threshold** (15 iterations) to find the sparsest gradient that fits within the client's budget. This simulates heterogeneous network conditions where different clients have different bandwidth constraints.

> **Note:** This project uses **LSTM** (Long Short-Term Memory), a classical recurrent neural network — not an LLM (Large Language Model).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python federated_lstm_budgeted.py
```

## License

MIT
