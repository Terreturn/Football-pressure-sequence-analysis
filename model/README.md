# spn

The sequence classifier uses 26 ordered structural pressure network inputs. The pooled XGBoost fit used 2,562,468 event rows from 1,245,248 pressing sequences across 9,220 matches. Each sequence had total training weight one. The public probability columns are `p_fail`, `p_neutral` and `p_success`, in that order.

| Artifact | Use |
| --- | --- |
| `model.ubj` | Pooled three-class sequence fit, 2,850 boosting rounds, maximum depth 7. |
| `temperature.json` | Transfer mapping fitted on held-out onsets. The inverse temperature is 0.9097933963724366. |
| `regain_0.ubj` through `regain_4.ubj` | Five 250-round, depth-3 binary fits for a controlled regain initiated within five seconds. Training used 20,293 observed pressures from 380 Premier League fixtures. |
| `regain_folds.json` | Original match ID to held-out fold. For other matches the scorer averages the five probabilities at each position. |
| `model_config.json` | Input order, class order, fit parameters, artifact checksums and inference settings. The sequence weight fingerprint is retained in exported metadata. |

Supply each match's original ID for positioning. It selects the fold and seeds the 64 candidate positions around the recorded presser. The score is the observed regain probability minus the mean feasible alternative probability, multiplied by 100.

The sequence artifact scores new matches. For internal cross-validation estimates on the training pool, use held-out predictions from training. [Feature definitions](../spn/MODEL.md)
