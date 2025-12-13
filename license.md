
---

## .gitignore

```gitignore
# ---- Python / venv ----
venv/
.venv/
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
*.egg-info/
dist/
build/
pip-wheel-metadata/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# ---- OS / Editor ----
.DS_Store
Thumbs.db
.vscode/
.idea/

# ---- Logs / outputs ----
*.log
runs/
outputs/
output/
metrics/
*.csv

# ---- Local datasets (DO NOT COMMIT) ----
dataset/
enrollments/
captures/
faces/
data/
*.npz

# ---- Model files (optional) ----
# If you want to keep model files out of git, ignore them.
# If you prefer to include YuNet in the repo, remove the yunet line below.
models/**/*.onnx
models/**/*.pt
models/**/settings.json

# ---- Secrets ----
.env
.env.*
*.key
*.pem

# ---- Media ----
*.mp4
*.mov
*.avi
*.mkv
*.png
*.jpg
*.jpeg
