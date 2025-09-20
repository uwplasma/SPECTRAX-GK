# Developer Guide

SPECTRAX-GK is a research-grade code, but designed with **modern Python development practices**.  
This page explains how to set up your environment and contribute.

---

## 🔹 Setup environment

```bash
git clone https://github.com/uwplasma/SPECTRAX-GK.git
cd SPECTRAX-GK
pip install -e ".[dev]"
pre-commit install
````

---

## 🔹 Workflow

### Code style & linting

We use [Ruff](https://github.com/astral-sh/ruff) as both linter and formatter.

```bash
ruff format .
ruff check .
```

### Type checking

Static analysis with [MyPy](https://mypy.readthedocs.io):

```bash
mypy .
```

### Testing

Run unit + regression tests:

```bash
pytest -v
```

Tests cover:

* Fourier & DG solvers
* Poisson operator
* Unit conversions
* Example configs (smoke tests)

✅ All three (`ruff`, `mypy`, `pytest`) must pass before merging.

---

## 🔹 Documentation

Build docs locally:

```bash
pip install mkdocs-material mkdocstrings[python] pymdown-extensions
mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
Docs auto-rebuild as you edit `docs/`.

---

## 🔹 Continuous Integration

* GitHub Actions run:

  * `ruff format --check`
  * `ruff check`
  * `mypy`
  * `pytest`
* Documentation is deployed to GitHub Pages at:
  [https://uwplasma.github.io/SPECTRAX-GK/](https://uwplasma.github.io/SPECTRAX-GK/)

---

## 🔹 Contributing

1. Fork the repo
2. Create a branch (`feature/my-new-feature`)
3. Add tests + docs
4. Run all checks locally (`ruff`, `mypy`, `pytest`)
5. Submit a Pull Request (PR)

---

## 📝 Tips

* Commit often, push early. CI will help you catch errors.
* Use the example configs as **regression tests** when adding features.
* Don’t be afraid to open a PR early with `[WIP]` — feedback is welcome.