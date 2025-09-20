# Installation

## From PyPI (recommended)

```bash
pip install spectraxgk
````

## From source (development)

```bash
git clone https://github.com/uwplasma/SPECTRAX-GK.git
cd SPECTRAX-GK
pip install -e ".[dev]"
pre-commit install
```

### Optional: JAX with GPU

Follow the official JAX instructions for your CUDA/CuDNN stack:
[https://github.com/google/jax#installation](https://github.com/google/jax#installation)
