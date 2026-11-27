# Structural Pressure Network

SPN uses StatsBomb events and 360 freeze frames to predict how a pressing sequence will end and to score a presser’s positioning. Follow one sequence from its first freeze frame through the 26 model inputs to its predicted outcome, then explore team and player results across all 64 matches of the 2022 FIFA World Cup.

**[View the World Cup sample workflow](tests/Sample_workflow.ipynb)** · [Run the notebook locally](#sample-workflow)

## Sample workflow

From the repository root, use Python 3.10 or newer in a virtual environment. On macOS, XGBoost requires the OpenMP runtime, available through `brew install libomp`.

```bash
python -m pip install -r requirements.txt
python -m jupyterlab tests/Sample_workflow.ipynb
```

Run the cells in order with an internet connection. `MATCH_IDS = None` selects every match with event and 360 coverage in the chosen competition and season. Set it to a list of match IDs for a smaller run. The default World Cup inputs occupy about 520 MiB. [StatsBomb Open Data](https://github.com/statsbomb/open-data)

### What the notebook produces

Match tables are saved under `output/sample/matches/<match_id>/`. Charts and their underlying CSVs are under `output/sample/plots/`.

| Result | Where to look |
| --- | --- |
| Sequences and predictions | `sequences.csv` has one row per detected press. `predictions.csv` gives calibrated failure, neutral and success probabilities at each scored anchor. |
| Model inputs and coverage | `features.csv` contains the 26 inputs in model order. `features_metadata.json` records their order and the model and file checksums. `result.json` includes match counts and exclusions. |
| Teams and players | `teams.csv` gives each team’s mean onset value, `p_success − p_fail`. `pressures.csv` scores each eligible named presser’s observed position against feasible alternatives, in percentage points of five-second regain probability. |
| Three rankings | Team press value, team positioning and player positioning across all 64 World Cup matches. Counts show how many scored sequences or pressures contribute. |
| Two team scatter plots | Press value versus xG difference per match uses all 64 fixtures. Mean positioning versus competition points uses the same 48 group-stage fixtures on both axes. |

The weights and configuration are in `model/`. Match processing, the two scoring tasks and plotting are implemented in `spn/`. [Model files and their roles](model/README.md) · [Definitions of the 26 features](spn/MODEL.md)

## Command-line use

The examples use the event and 360 JSON downloaded by the notebook. Substitute paths to your own paired StatsBomb files as needed.

**Score one match** and export its features.

```bash
python run.py --events output/sample/events/3869685.json --three-sixty output/sample/three-sixty/3869685.json --task both --export-features --output output/final
```

**Score a batch.** The two input directories are paired by filename stem. Use original match IDs for model fold selection. For renamed single-match files, supply `--match-id`.

```bash
python run.py --events-dir output/sample/events --three-sixty-dir output/sample/three-sixty --task both --output output/tournament
```

The default task is `sequence`. Choose `positioning` for individual presser scores or `both` for both tasks. Python callers can import `run_pipeline` from `run`.

### Plot match results

**Plot existing scores** using one match directory or the batch output. The event files supply team and player names and shot xG.

```bash
python plot.py --input output/tournament --events-dir output/sample/events --output plots/tournament
```

Use `python run.py --help` and `python plot.py --help` for the available arguments. The notebook demonstrates how to select the fixtures appropriate to each scatter plot.

## Contributors and licence

[Brennan Klein](https://github.com/jkbren) and [Maddalena Torricelli](https://github.com/maddaleona) contributed code. [Terreturn](https://github.com/Terreturn) integrated and published their contributions.

The repository’s MIT licence covers the released code and model-pipeline packaging. The bundled shape-graph implementation includes its own attribution and licence. StatsBomb data is subject to its own terms. [Repository licence](LICENSE) · [Shape-graph licence](spn/vendor/LICENSE) · [StatsBomb Open Data](https://github.com/statsbomb/open-data)
