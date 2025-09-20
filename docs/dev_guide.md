
# Developer Guide

## Environment
```bash
git clone https://github.com/uwplasma/SPECTRAX-GK.git
cd SPECTRAX-GK
pip install -e ".[dev]"
pre-commit install
````

## Workflow

* **Format/Lint**: `ruff format . && ruff check .`
* **Types**: `mypy .`
* **Tests**: `pytest -v`

Pre-commit hooks run these checks automatically on commit.

## Docs locally

```bash
pip install mkdocs-material mkdocstrings[python] pymdown-extensions
mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Contributing

1. Create a feature branch (e.g., `feature/new-thing`)
2. Add tests & docs
3. Run all checks locally
4. Submit a PR
