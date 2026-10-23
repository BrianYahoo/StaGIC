# StaGIC runtime environment

This directory defines the runtime for the modeling, optimization, validation,
and visualization code in `code/script/`. Run all installation commands below
from the **repository root**.

## Create the environment

```bash
conda env create -f requirements/environment.yml
conda activate StaGIC
```

The environment pins Python 3.9.25, BrainPy 2.6.0, JAX 0.4.28, and
`jaxlib==0.4.28+cuda12.cudnn89`. It includes the JAX CUDA wheel index required by
that backend. Use a compatible NVIDIA GPU and CUDA runtime for training.

## Dependency files

| File | Purpose |
| :--- | :--- |
| `environment.yml` | Conda environment named `StaGIC`, including the CUDA-enabled JAX backend. |
| `requirements.txt` | Direct runtime dependency pins for an existing environment; install an appropriate JAX backend separately. |
| `requirements-lock.txt` | Direct dependencies plus key backend packages from the source runtime, including the pinned CUDA-enabled `jaxlib`. |

The lock file records key runtime pins; it is not a complete, platform-independent
lock of every transitive dependency.

For an existing compatible Python environment, choose one installation route:

```bash
# Direct dependencies; backend configuration is managed separately.
python -m pip install -r requirements/requirements.txt
```

```bash
# CUDA backend and key runtime pins matching the release.
python -m pip install -r requirements/requirements-lock.txt
```

## Verify installation

```bash
python -c "import brainpy, jax, numpy, pandas, matplotlib, seaborn, tqdm; print('BrainPy:', brainpy.__version__); print('JAX:', jax.__version__); print(jax.devices())"
python -c "import sys; sys.path.insert(0, 'code/script'); import func_settings, running; print('StaGIC imports OK')"
```

Check that `jax.devices()` lists a GPU before starting training. The training
implementation explicitly transfers arrays to CUDA; selecting CPU at the
entry point does not make training CPU-compatible. Validation and testing use
the CPU through `post.sh` by default. A CPU-only environment can use
`jaxlib==0.4.28` in place of the CUDA wheel for imports and CPU evaluation.

On a server without a display, set `MPLBACKEND=Agg` when running postprocessing.
The [main README](../README.md#quick-start) documents the complete workflow,
device selection, and seed controls.
