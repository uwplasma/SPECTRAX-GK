#!/usr/bin/env python3
"""Validate release-version consistency before publishing artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import urllib.request
from typing import Iterable

try:  # pragma: no cover - Python 3.10 fallback
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[2]
SOURCE_VERSION = ROOT / "src" / "spectraxgk" / "_version.py"
PYPROJECT = ROOT / "pyproject.toml"
VERSION_RE = re.compile(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]\s*$")


class ReleaseVersionError(ValueError):
    """Raised when release metadata is internally inconsistent."""


def read_project_version(root: Path = ROOT) -> str:
    """Return the PEP 621 project version from ``pyproject.toml``."""

    path = root / "pyproject.toml"
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        version = payload["project"]["version"]
    except KeyError as exc:
        raise ReleaseVersionError("pyproject.toml is missing project.version") from exc
    if not isinstance(version, str) or not version.strip():
        raise ReleaseVersionError(
            "pyproject.toml project.version must be a non-empty string"
        )
    return version.strip()


def read_source_version(root: Path = ROOT) -> str:
    """Return ``spectraxgk.__version__`` without importing the package."""

    path = root / "src" / "spectraxgk" / "_version.py"
    for line in path.read_text(encoding="utf-8").splitlines():
        match = VERSION_RE.match(line.strip())
        if match:
            return match.group(1)
    raise ReleaseVersionError(f"{path.relative_to(root)} does not define __version__")


def normalize_tag(tag: str | None) -> str | None:
    """Normalize GitHub tag strings such as ``refs/tags/v1.2.3``."""

    if tag is None:
        return None
    tag = tag.strip()
    if not tag:
        return None
    if tag.startswith("refs/tags/"):
        tag = tag.removeprefix("refs/tags/")
    return tag


def default_tag_from_github_env() -> str | None:
    """Return the GitHub ref name only for tag-triggered workflows.

    Branch pushes expose ``GITHUB_REF_NAME`` as values such as ``main``. Treating
    that as a release tag would make normal CI fail, so branch workflows must
    opt in with an explicit ``--tag`` if they need tag validation.
    """

    if os.environ.get("GITHUB_REF_TYPE") != "tag":
        return None
    return os.environ.get("GITHUB_REF_NAME")


def fetch_pypi_versions(package: str) -> set[str]:
    """Return released versions for ``package`` from the public PyPI JSON API."""

    url = f"https://pypi.org/pypi/{package}/json"
    with urllib.request.urlopen(url, timeout=20) as response:  # noqa: S310 - fixed PyPI HTTPS endpoint
        payload = json.loads(response.read().decode("utf-8"))
    releases = payload.get("releases", {})
    if not isinstance(releases, dict):
        raise ReleaseVersionError(f"PyPI response for {package!r} is missing releases")
    return {str(version) for version in releases}


def validate_release_version(
    *,
    root: Path = ROOT,
    tag: str | None = None,
    require_tag: bool = False,
    package: str = "spectraxgk",
    pypi_versions: Iterable[str] | None = None,
) -> dict[str, object]:
    """Validate package version, source version, optional tag, and PyPI uniqueness."""

    root = root.resolve()
    project_version = read_project_version(root)
    source_version = read_source_version(root)
    if source_version != project_version:
        raise ReleaseVersionError(
            f"src/spectraxgk/_version.py has {source_version!r}, "
            f"but pyproject.toml has {project_version!r}"
        )

    normalized_tag = normalize_tag(tag)
    if require_tag and normalized_tag is None:
        raise ReleaseVersionError("release publishing requires a tag like v1.2.3")
    if normalized_tag is not None:
        expected = f"v{project_version}"
        if normalized_tag != expected:
            raise ReleaseVersionError(
                f"release tag {normalized_tag!r} does not match project version {project_version!r}; "
                f"expected {expected!r}"
            )

    duplicate_on_pypi = False
    if pypi_versions is not None:
        duplicate_on_pypi = project_version in {
            str(version) for version in pypi_versions
        }
        if duplicate_on_pypi:
            raise ReleaseVersionError(
                f"{package} {project_version} already exists on PyPI; bump the version before publishing"
            )

    return {
        "package": package,
        "project_version": project_version,
        "source_version": source_version,
        "tag": normalized_tag,
        "require_tag": require_tag,
        "checked_pypi": pypi_versions is not None,
        "duplicate_on_pypi": duplicate_on_pypi,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root.")
    parser.add_argument(
        "--tag",
        default=default_tag_from_github_env(),
        help="Release tag to validate.",
    )
    parser.add_argument(
        "--require-tag",
        action="store_true",
        help="Fail unless --tag is a v-prefixed release tag.",
    )
    parser.add_argument(
        "--check-pypi",
        action="store_true",
        help="Fail if this version already exists on PyPI.",
    )
    parser.add_argument(
        "--package",
        default="spectraxgk",
        help="PyPI package name for duplicate checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pypi_versions = fetch_pypi_versions(args.package) if args.check_pypi else None
    try:
        report = validate_release_version(
            root=args.root,
            tag=args.tag,
            require_tag=bool(args.require_tag),
            package=str(args.package),
            pypi_versions=pypi_versions,
        )
    except ReleaseVersionError as exc:
        raise SystemExit(f"release-version check failed: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
