# Installation

SPECTRAX-GK is a pure-Python package built on [JAX](https://github.com/google/jax).  
You can install it in three main ways: from PyPI, from source, or with GPU support.

---

## 🔹 Install from PyPI (recommended)

For most users:

```bash
pip install spectraxgk
```

This installs the core package with CPU-based JAX. It works out of the box on Linux, macOS, and Windows (WSL2 recommended).

---

## 🔹 Install from source (development mode)

If you plan to contribute, run tests, or edit the code:

```bash
git clone https://github.com/uwplasma/SPECTRAX-GK.git
cd SPECTRAX-GK
pip install -e ".[dev]"
pre-commit install
```

* `-e` makes the install *editable* (local changes take effect immediately).
* `.[dev]` pulls in developer extras (pytest, ruff, mypy, docs tools).
* `pre-commit install` ensures style & tests run before every commit.

---

## 🔹 JAX with GPU/TPU acceleration

By default, `pip install spectraxgk` installs **CPU-only JAX**.
For GPUs or TPUs, follow the official [JAX installation guide](https://github.com/google/jax#installation).

Example for CUDA 12:

```bash
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Replace `cuda12` with the CUDA version on your machine.

---

## 🔹 Verify installation

Check that the CLI is available:

```bash
spectraxgk --help
```

Run the built-in two-stream example:

```bash
spectraxgk --input examples/two_stream.toml
```

✅ You should see a printed simulation summary and plots for energy, fields, and distributions.

---

## 🔹 Optional: Development extras

* **Docs**:

  ```bash
  pip install mkdocs-material mkdocstrings[python] pymdown-extensions
  mkdocs serve
  ```

  Then visit [http://127.0.0.1:8000](http://127.0.0.1:8000).

* **Testing**: `pytest -v`

* **Lint/Format**: `ruff format . && ruff check .`

* **Types**: `mypy .`