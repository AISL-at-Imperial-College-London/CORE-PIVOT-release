# Environment setup

The default is `uv sync --locked`, followed by `uv run --locked python ...` from
the repository root. uv selects Python 3.12 and installs the locked CPU reviewer
environment, including plotting/demo dependencies. Neither system Python nor
shell activation is needed. These commands do not start training.

## Linux / macOS

Install uv with `brew install uv` on macOS, or the official standalone installer:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reopen the terminal, enter the repository folder and run:

```sh
uv sync --locked
uv run --locked python run.py --help
```

Use uv 0.13.7 or newer. The macOS PyTorch build targets Apple Silicon.
See [official installation alternatives](https://docs.astral.sh/uv/getting-started/installation/).

## Windows troubleshooting

Start with `.\setup.cmd`, then `.\review.cmd --help`. These batch files do not
require Python or uv on PATH and do not use PowerShell scripts. They work from
the repository's location even if invoked from another directory. Setup installs
dependencies and checks imports only. `review.cmd` passes your arguments to
`run.py`; explicit `train` commands do train, whereas `--dry-run` does not.
For lower-level scripts, use `.\review.cmd python scripts/SCRIPT.py ...`.

The wrappers look for uv at the WinGet user link, within uv's user/machine WinGet
package directories when the link is missing, the user's `.local/bin` and
`.cargo/bin`, the machine WinGet link, and finally PATH. Setup installs missing uv
via WinGet's explicit community source and repeats discovery. It does not accept
agreements automatically or change execution policy, PATH, or security settings.

If WinGet is absent or your standalone installation is elsewhere, download and
extract the official [Windows uv release](https://github.com/astral-sh/uv/releases).
Point to its executable for this PowerShell session (including paths with spaces):

```powershell
$env:CORE_PIVOT_UV = 'C:\path to extracted uv\uv.exe'
.\setup.cmd
.\review.cmd --help
```

Use uv 0.13.7 or newer. A bad override produces an error rather than silently
using another installation. Locked sync enforces the required uv version.
On institution-managed machines, ask your administrator if installation is blocked.
Do not change execution policy or disable security software. Transient file-lock
errors may resolve by retrying setup after it exits; persistent access or network
errors require local administrator/proxy configuration. Do not copy `.venv` between
machines. Update the checkout rather than regenerating a mismatched lockfile.

## Optional CUDA training

The default uv project is deliberately CPU-only. Do not simply append
`--device cuda` to its commands. Use a separate environment for NVIDIA training:

```sh
uv venv --python 3.12 .venv-cuda
```

Windows PowerShell:

If uv is not on PATH, use the full executable path printed by `setup.cmd` in place
of `uv` below (PowerShell syntax: `& 'C:\path\uv.exe' ...`).

```powershell
uv pip install --python .venv-cuda/Scripts/python.exe torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-cuda/Scripts/python.exe -e ".[figures,demo]"
.\.venv-cuda\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Linux:

```sh
uv pip install --python .venv-cuda/bin/python torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-cuda/bin/python -e ".[figures,demo]"
.venv-cuda/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

This separate CUDA environment is **not** the locked CPU environment. CUDA 13
requires a compatible NVIDIA driver; choose another supported PyTorch backend
if necessary using the [official guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
Record its installed versions when comparing runs. macOS does not support CUDA.
Use that environment's explicit Python path for commands in the README, in place
of `uv run --locked python`, and set `--device cuda` only when deliberately training.
Do not run project `uv sync` inside `.venv-cuda` or overwrite the CPU environment.

## Optional pip setup

If uv cannot be installed, Python 3.12 and pip remain supported, without activation:

Windows:

```powershell
py -V:3.12 -m venv .venv-pip
.\.venv-pip\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-pip\Scripts\python.exe -m pip install -e ".[figures,demo]"
```

Linux/macOS:

```sh
python3.12 -m venv .venv-pip
.venv-pip/bin/python -m pip install -e ".[figures,demo]"
```

Use the explicit `.venv-pip` interpreter thereafter. This fallback does not enforce
`uv.lock`; it may resolve different versions. The locked uv workflow is preferred.
Historical runtime versions in `requirements-reference.txt` document the paper's
experiments, not a guarantee of identical numerical outcomes on other hardware.
