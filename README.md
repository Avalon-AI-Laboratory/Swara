# Swara TTS

## Installation

Install `uv` package manager.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the virtual environment using `uv` and install the project dependencies.

```bash
uv venv -p 3.13
```

Activate virtual environment.

```bash
# Mac/Linux
source .venv/bin/activate

# Windows
.venv/Scripts/activate
```

Install project dependencies.

```bash
uv pip install -e .
```

Or

```bash
uv sync
```

Create `.env` to store token credentials for `wandb`

```bash
WANDB_API_KEY=
```
