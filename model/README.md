# spn

The public `spn` release supports two tasks. **Sequence prediction** estimates the outcome of a pressing sequence at each eligible event. **Player positioning** compares the predicted chance of regaining the ball at the presser's observed position with the chances at feasible nearby positions.

## Files used for each task

| Mode in `run.py` | Files loaded from this directory |
| --- | --- |
| `--task sequence` (default) | `model_config.json`, `model.ubj` and `temperature.json` (3 files). |
| `--task positioning` | `model_config.json`, `regain_folds.json` and either one regain weight file for a match ID in the fold map (3 files total) or all five for any other match ID (7 files total). |
| `--task both` | Both sets above, with one shared config (5 files for a match ID in the fold map, 9 for any other match ID). |

### Shared configuration and documentation

- `model_config.json` identifies the weights and calibration file and stores the SHA-256 checksums used to verify them when loaded. It also records the order of the 26 input features and three output classes, the pressure and sequence selection settings, the label thresholds and the training parameters for both tasks. Exported sequence metadata includes the weight fingerprint. [Definitions of the 26 inputs](../spn/MODEL.md)
- `README.md` is this guide for readers. The inference code reads the configuration and weights listed above.

### Sequence prediction

- `model.ubj` contains the pooled XGBoost classifier. It takes the 26 features for an eligible event and produces three raw probabilities for the eventual sequence outcome. Training used 2,562,468 event rows from 1,245,248 pressing sequences across 9,220 matches, with total weight one per sequence, 2,850 boosting rounds and a maximum tree depth of 7. Use this fit to score new matches and the original held-out predictions to evaluate cross-validation performance on the training pool.
- `temperature.json` converts those raw probabilities into the calibrated values exported to `predictions.csv` as `p_fail`, `p_neutral` and `p_success`. The inverse temperature (0.9097933963724366) was fitted on held-out sequence onsets.

### Player positioning

- `regain_0.ubj`, `regain_1.ubj`, `regain_2.ubj`, `regain_3.ubj` and `regain_4.ubj` each predict the probability that the pressing team initiates a controlled regain within five seconds. The five fits use 20,293 observed pressures from 380 Premier League matches split into five groups. The number in each filename identifies the group left out when fitting that model. Each fit used 250 boosting rounds and a maximum tree depth of 3.
- `regain_folds.json` maps original match IDs to their held-out group. For example, if a match maps to group 2, the scorer uses `regain_2.ubj`. For an unlisted match it averages all five predictions **at each position**. Supply the original match ID so the scorer can select the weights and, together with the event ID, reproduce the candidate positions.

For an eligible high press recorded by StatsBomb as a `Pressure` event with an identified presser, the scorer evaluates the player's observed position and samples 64 nearby candidates between 0.5 and 2 metres away, keeping the other players fixed. It keeps only feasible alternatives and writes a row to `pressures.csv` when at least one is available. The score `position_quality_pp` is 100 times the difference between the observed regain probability and the mean probability across feasible alternatives. For example, an observed probability of 0.32 and an alternative mean of 0.28 give a score of **+4 percentage points**.
