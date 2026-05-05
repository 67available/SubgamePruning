# Subgame Pruning for Two-Player Imperfect-Information Games

This repository contains the Rust implementation accompanying the paper:

> Shenkai Zhang, Zekeng Zeng, Peipei Yang, and Junge Zhang.  
> **Subgame Pruning: Efficiently Solving Two-Player Imperfect Information Games by Accelerated Public Tree Traversal in CFR**.  
> *IEEE Transactions on Games*, 2026.  
> DOI: [10.1109/TG.2026.3675377](https://doi.org/10.1109/TG.2026.3675377)

The code implements vanilla CFR/CFR+ and their subgame-pruning variants for two-player imperfect-information games represented as public trees.

## Overview

The repository currently includes:

- `cfr`: baseline CFR / CFR+
- `sp`: Partial SP
- `sp-general`: Two-stage SP
- `spc`: Partial SP, where pruning is triggered only after the pruning condition is satisfied for `k` consecutive checks
- `sp-general-c`: Two-stage SP, where pruning is triggered only after the pruning condition is satisfied for `k` consecutive checks

The executable reads a pre-generated game tree from `tree/<env>/trees_<env>_0923.txt`, runs training, and stores metrics under `experiments_13700/`.

## Repository Structure

- `src/main.rs`: CLI entry point
- `src/cfr.rs`: baseline CFR / CFR+
- `src/subgame_prune_check_free_only.rs`: partial subgame pruning
- `src/subgame_prune_general.rs`: two-stage subgame pruning
- `src/subgame_prune_check_free_only_c.rs`: partial subgame pruning with consecutive-condition triggering
- `src/subgame_prune_general_c.rs`: two-stage subgame pruning with consecutive-condition triggering
- `src/br.rs`: best response used for exploitability evaluation
- `src/load_tree.rs`: game-tree loading
- `src/logger.rs`: experiment logging
- `tree/`: pre-generated public trees / histories
- `py_scripts/`: plotting and summary scripts
- `run_*.ps1`: Windows PowerShell scripts for batch experiments

## Requirements

- Rust toolchain, recommended via [rustup](https://rustup.rs/)
- Python 3.x for plotting and post-processing scripts
- Windows PowerShell for the provided batch scripts

Python scripts rely on common scientific packages such as:

- `pandas`
- `matplotlib`
- `numpy`

## Usage

Before running any experiment, extract [tree.zip](./tree/tree.zip) into the `tree/` directory.

Basic command format:

```powershell
cargo run -- <exp_name> <purpose...> --env <env> --epoch <N> --algo <algo> [--cfr-plus] [--rand-init]
```

Arguments:

- `<exp_name>`: experiment directory name under `experiments_13700/`
- `<purpose...>`: free-form description recorded in `purpose.txt`
- `--env`: game name
- `--epoch`: number of training iterations
- `--algo`: one of `cfr`, `sp`, `sp-general`, `spc`, `sp-general-c`
- `--cfr-plus`: enable CFR+
- `--rand-init`: random initialization of policy

## Quick Examples

Run vanilla CFR on Leduc Poker:

```powershell
cargo run -- "cfr/leduc_poker" "baseline CFR on leduc poker" --env leduc_poker --epoch 1000000 --algo cfr
```

Run two-stage subgame pruning on Leduc Poker:

```powershell
cargo run -- "cfr-sp-general/leduc_poker" "two-stage SP on leduc poker" --env leduc_poker --epoch 1000000 --algo sp-general
```

Run CFR+ with compensated two-stage subgame pruning:

```powershell
cargo run -- "cfr-plus-sp-general_c/leduc_poker" "CFR+ with compensated two-stage SP" --env leduc_poker --epoch 1000000 --algo sp-general-c --cfr-plus
```

## Available Environments

The repository currently ships tree files for:

- `leduc_poker`
- `leduc5-3-2`
- `liars_dice`
- `tic_tac_toe`
- `tiny_bridge_2p`

Each environment should have a corresponding tree file:

```text
tree/<env>/trees_<env>_0923.txt
```

If you want to add a new game, you need to generate the serialized public-tree / infoset / history file in the same format expected by `src/load_tree.rs`.

## Output Format

Each run creates a directory like:

```text
experiments_13700/<exp_name>/<date>_exp-XXX/
```

Typical outputs:

- `purpose.txt`: experiment description and command line
- `metrics.csv`: logged metrics over training

## Reproducing the Paper Experiments

The repository includes ready-to-edit PowerShell launchers:

- `run_leduc.ps1`
- `run_leduc5.ps1`
- `run_liar_dice.ps1`
- `run_bridge.ps1`
- `run_ttt.ps1`
- `run_all.ps1`

These scripts were used to launch multiple experiments, often with CPU affinity binding on Windows.

### Step 1: Build the release binary

```powershell
cargo build --release
```

### Step 2: Run one environment

For example, on Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_leduc.ps1
```

### Step 3: Aggregate or visualize results

Example plotting command:

```powershell
python py_scripts/plot_exploit.py --output_dir ./plots_exploit_pruning --logx
```

Example summary scripts:

```powershell
python py_scripts/raw_avg_time.py
python py_scripts/gather_avg_time.py
python py_scripts/gather_pruning_prop.py
```

Notes:

- Some Python scripts assume the base results directory is `./experiments_13700`.
- `raw_avg_time.py` generates `metrics_f.csv`, which is then consumed by `gather_avg_time.py`.
- The scripts are research utilities and may require small path adjustments depending on your local run layout.

## Citation

If you use this code, please cite:

```bibtex
@ARTICLE{11441460,
  author={Zhang, Shenkai and Zeng, Zekeng and Yang, Peipei and Zhang, Junge},
  journal={IEEE Transactions on Games},
  title={Subgame Pruning: Efficiently Solving Two-Player Imperfect Information Games by Accelerated Public Tree Traversal in CFR},
  year={2026},
  volume={},
  number={},
  pages={1-9},
  doi={10.1109/TG.2026.3675377}
}
```

## Notes

- This repository assumes pre-generated game trees rather than building them online during training.
- This repository implements only the full-traversal DFS version of CFR/subgame pruning; Monte Carlo sampling variants are not included.
