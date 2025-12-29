# CNN (Text) + Custom Automatic Differentiation in Julia

This repository contains a **from-scratch** implementation of:
- a small **reverse-mode Automatic Differentiation** engine (`AD.jl`)
- a minimal **neural network library** built on top of it (`NETWORK.jl`)
- an example **text CNN for IMDb sentiment classification** (`CNN.ipynb`)
- **reference implementations** in PyTorch and TensorFlow for comparison (`Reference_models/`)

The goal is not feature completeness, but a transparent, inspectable baseline for:
- building computation graphs
- forward/backward passes
- training loops and optimizers
- basic CNN building blocks for text

## Repository contents

- `AD.jl` — core AD engine (graph nodes, ops, forward/backward, conv/pooling primitives)
- `NETWORK.jl` — layers + training utilities (Embedding, Conv1D, MaxPool1D, Flatten, Dense, Chain, Adam/SGD, Trainer)
- `CNN.ipynb` — example training run for IMDb (5 epochs)
- `Reference_models/`
  - `PyTorch_cnn.py`
  - `TensorFlow_cnn.py`
- `AWID_Jankowicz_Walczak_KM3.pdf` — report (design + results + comparisons)

## Requirements

### Julia
- Julia (tested with standard libraries)
- For running the notebook: `JLD2` (and optionally `BenchmarkTools`)

Install:
```julia
import Pkg
Pkg.add(["JLD2", "BenchmarkTools"])
```

# Collaborators

Collaborators for this repository include:
* Patryk Jankowicz ([GitHub](https://github.com/PatrykSJ)), Warsaw University of Technology
* Jan Walczak ([GitHub](https://github.com/JanWalczak)), Warsaw University of Technology




