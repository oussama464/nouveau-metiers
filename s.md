# UV Proof-of-Concept (PoC) User Guide

## Overview

**UV** is a modern Python dependency management and task runner tool that combines the functionality of `pip`, `build`, `twine`, and virtual environment management into a single interface. Unlike traditional tools:

- **pip** installs packages but doesn’t manage virtual environments or lock files.
- **poetry** offers lockfile support and environment isolation but can be slower for editable installs and CI workflows.
- **UV** provides:
  - **Ultra‑fast synchronization** of dependencies into a venv
  - **Editable installs** out of the box (`uv sync`)
  - **Built‑in build & publish** commands (`uv build`, `uv publish`)
  - **Lockfile support** for reproducible CI  
  - **Zero‑touch venv activation** via `uv run`

In our PoC, the editable‑mode install of our `imad` package using traditional workflows was a bottleneck. By switching to **UV**, we benefit from faster setup and streamlined CI steps.

---

## Why UV vs. pip & Poetry

| Feature                  | pip + venv       | Poetry           | UV                        |
| ------------------------ | ---------------- | ---------------- | ------------------------- |
| Editable install (`-e`)  | Manual `pip install -e .` | `poetry install --editable` (slower) | `uv sync` (optimized) |
| Lockfile management      | n/a              | ✔️               | ✔️                        |
| Build & publish          | `python -m build` + `twine upload` | `poetry build && poetry publish` | `uv build --wheel` + `uv publish` |
| Virtual environment mgmt | `python -m venv` / `venv/bin/activate` | Automatic venv creation | Automatic `.venv` or use existing |
| CI friendliness          | Multiple tools   | Multiple steps   | Single-tool, lock-based  |

---

## Local User Guide

### 1. Install UV

```bash
pip install uv
```

### 2. Initialize / Sync Dependencies
UV can either create its own .venv or use your existing venv310
* Create & activate a new .venv:
```bash
uv sync
# → Creates `.venv/`, installs locked deps, and installs `imad` in editable mode
source .venv/bin/activate
```
uv sync does the following in a single workflow:

    * Dependency resolution & locking
    Reads your project’s pyproject.toml, resolves versions, writes (or updates) uv.lock.

    * Virtual-env setup
    Creates a new .venv (or, with --active, uses your existing venv) and ensures it has exactly the locked packages.

    * Editable install
    Installs your local project (e.g. imad) in editable mode so you can iterate without reinstalling.

Together, these steps keep your venv and lockfile perfectly in sync and ready to run.

* Use the existing venv310:
```bash
deactivate       # leave any active venv
uv sync --active
# → Detects `venv310/`, installs deps in place, and installs `imad` editably
```

### 3. Managing Dependencies
* Add a dependency to your project

```bash
uv add <package-name>
```
Installs <package-name> and updates uv.lock.

* Add a dev/test dependency
```bash
uv add <package-name> --group dev
uv add <package-name> --group test
```

* Remove a dependency

```bash
uv add <package-name> --group dev
uv add <package-name> --group test
```

**Note**:
All of uv add / uv sync / uv remove automatically update uv.lock. Commit uv.lock alongside your code

### 4. Running Scripts & Tests

you can activate the .venv created by uv and use as usual python script.py 
or No need to activate your venv manually. Simply run:

```bash
uv run python script.py
```
UV will automatically select the right environment.



## CI Integration

### Install & sync dependencies (locked):

```bash
uv sync --locked
```

### Build wheel:

```bash
uv build --wheel
```
### Publish to Artifactory:

```bash
uv publish
```

No more separate pip, build, twine ... steps — UV handles it all.


## cache managment / maintenance

- **`uv cache clean`**  
  Remove **all** cache entries (or pass a package name to remove only that package’s cache).

- **`uv cache prune`**  
  Delete **unused**/outdated entries (e.g. old buckets from prior uv versions). Safe to run periodically.

- **`uv cache prune --ci`**  
  CI-optimized prune: removes downloaded wheels and source dists, but keeps any wheels you built from source.

- **`uv cache dir`**  
  Print the path to the active cache directory.


- **`uv sync --refresh`**  
  Revalidate *all* cached data before syncing.

- **`uv sync --refresh-package <name>`**  
  Revalidate cache for a single package.

- **`uv sync --reinstall`**  
  Ignore any already-installed versions and reinstall everything.


- **`pip install --upgrade uv`**  
  upgrade uv.

- **`uv tool dir`**  
  Show where uv keeps its “tool” (plugin) installations.


## Performance Comparison
