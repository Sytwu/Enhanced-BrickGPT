# Development Notes

## Setup (one-time)

```bash
pip install uv                  # if uv missing
huggingface-cli login           # needs Llama-3.2-1B-Instruct access, try 'hf auth login' if didn't work
uv venv
uv sync                         # add --extra finetuning if needed
git submodule update --init
```

Download LDraw parts library into the repo (shared across hosts):

```bash
wget https://library.ldraw.org/library/updates/complete.zip                 # try "curl -O https://library.ldraw.org/library/updates/complete.zip" in windows w/ git bash
unzip complete.zip && rm complete.zip   # creates ./ldraw/                        "unzip complete.zip -d ldraw"
```

Download background exr (Google Drive) into `ImportLDraw/loadldraw/`:

```bash
pip install gdown
gdown 1Yux0sEqWVpXGMT9Z5J094ISfvxhH-_5K -O ImportLDraw/loadldraw/background.exr
```

## Usage (every session)

```bash
source .venv/bin/activate                                                   # source .venv/Scripts/activate
export LDRAW_LIBRARY_PATH="$(pwd)/ldraw"
export GRB_LICENSE_FILE="$(pwd)/gurobi.lic"   # optional, for Gurobi
```

## Inference

```bash
uv run infer                    # interactive
uv run infer --use_gurobi False # no Gurobi license
uv run infer -h                 # all options
```

Prompts asked: text → output filename → seed.

## Outputs

Saved next to the given filename:

- `output.png` — rendered image
- `output.txt` — brick-by-brick text
- `output.ldr` — LDraw format

Absolute paths print to stdout. Open with `xdg-open output.png`.

## Texture / Mesh2Brick

```bash
cd src/texture     # see local README
cd src/mesh2brick  # see local README
```

## Fine-tuning

```bash
uv sync --extra finetuning
uv run prepare_finetuning_dataset --input_path AvaLovelace/StableText2Brick --output_path <DATA>
uv run accelerate config
uv run ./scripts/finetune.zsh <PRETRAINED_DIR> <OUTPUT_DIR> <RUN_NAME> <DATA>
```
