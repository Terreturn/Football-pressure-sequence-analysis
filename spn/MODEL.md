# Model and selected features

This note describes the frozen model shipped in `../model/`. The source of truth
for feature names, their **input order**, and preprocessing settings is
[`model_config.json`](../model/model_config.json); the calculations are in
[`feature_engineering.py`](feature_engineering.py) and [`_network.py`](_network.py).

## Model

- **Identity:** `current-spn-top19-xgb-raw`, version `1.0.0`.
- **Task:** predict `fail`, `neutral`, or `success` for each complete high-pressure
  sequence anchor. The output is the model's **raw** three-class probability;
  no isotonic calibration is applied.
- **Estimator:** M1 sequence-weighted XGBoost (`multi:softprob`), saved as a
  native UBJ model. Each training sequence had total weight 1, distributed as
  `1 / anchor_count` across its anchors. The 19 input features are the frozen
  `SPN_weighted_top_19` subset; outcome labels are never model inputs.
- **Feature selection:** the 19 inputs were chosen from 29 candidate SPN
  features using training-match validation and a one-standard-error subset
  rule. The public release contains the selected inference contract.
- **Training settings:** 500 boosting rounds, maximum depth 6, learning rate
  0.03, row sampling 0.8, feature sampling per tree 0.8, minimum child weight
  5, split penalty (`gamma`) 1, and L2 penalty (`reg_lambda`) 3. These settings
  describe the trained artifact; inference loads the saved trees rather than
  retraining them. The public repository does not include training code.

## Frozen preprocessing parameters

| Component | Main settings |
| --- | --- |
| Coordinates | StatsBomb `120 × 80` converted to a `105 × 68 m` pitch; play is oriented in the attacking direction. |
| Player pressure | Logistic distance kernel: centre `6.4 m`, spread `2.2 m`. |
| Boundary pressure | Centre `3.2 m`, spread `1.1 m`. |
| Passing lane | Centre `1.2 m`, spread `0.6 m`; pass-distance decay `18 m`; open-pass weight threshold `0.5`. |
| Network edges | Minimum pressure/edge weight `0.05`. |
| Sequence detection | Pressure threshold `0.65`, own-half limit `x < 60` on the StatsBomb pitch, maximum anchor gap `5 s`. |
| Outcome rules | Minimum displacement `3 m`, long switch `25 m`, high clearance/pass `35 m`, receiver-relief threshold `0.65`, forward/backward angle cutoffs `60° / 120°`. |

The manifest also fixes smaller context-matching tolerances; consult
[`model_config.json`](../model/model_config.json) for their exact values.
Changing these settings while reusing the frozen model changes its input
meaning.

## The 19 selected features

Categories describe the **primary source** of each feature. Geometry and
network structure overlap: SPN edges and Voronoi regions are built from player
locations in the 360 freeze frame. The groups below contain 14 anchor-level
features and five changes between anchors.

### Direct geometry and incoming-ball context (3)

| Feature | Meaning |
| --- | --- |
| `carrier_x_norm` | Attack-normalised carrier x coordinate (`0–1`) from the freeze-frame actor. |
| `ball_in_dist` | Metric distance travelled by the incoming ball from the resolved sequence context. |
| `ball_in_angle_cos` | Cosine of that incoming movement's angle. |

### Geometry-based pressure and pitch boundaries (4)

| Feature | Meaning |
| --- | --- |
| `P_total` | Combined defender-to-carrier pressure, `1 − ∏(1 − p_i)`, from distance weights. This **feature excludes boundary pressure**, although the Stage-1 detector can include it. |
| `effective_pressers` | Effective number of active defender pressure contributions, `(Σp_i)² / Σp_i²`. |
| `weighted_angular_dispersion` | How widely defenders surround the carrier, weighted by their pressure. |
| `n_active_carrier_boundaries` | Count of pitch-boundary edges affecting the carrier at the network threshold. |

### SPN pressure, passing and Voronoi network readouts (7)

| Feature | Meaning |
| --- | --- |
| `mean_receiver_pressure` | Mean summed defender pressure on visible attacking receivers. |
| `press_target_entropy` | Entropy of pressure allocated across the carrier and attacking receivers. |
| `pressure_mean_focus_entropy` | Mean defender-level entropy of pressure-edge weights across targets. |
| `best_forward_pass_w` | Highest weighted carrier-to-receiver pass edge with the receiver ahead of the carrier. |
| `n_open_pass` | Number of pass edges with weight **greater than** `0.5`; this counts all directions, not only forward passes. |
| `vor_carrier_area_share` | Carrier Voronoi-cell area divided by the camera-visible pitch area. |
| `vor_forward_receiver_area_share` | Sum of Voronoi-cell areas for receivers ahead of the carrier, divided by the camera-visible pitch area. |

Pass-edge weight combines distance decay and the strongest defender's passing
lane block. The Voronoi shares require a usable 360 `visible_area`; if it is
unavailable, these two features are missing. Delaunay graph features are
computed among the broader candidates but are **not** in this frozen subset.

### Within-sequence changes (5)

| Feature | Source feature |
| --- | --- |
| `d_carrier_x_norm_dt` | `carrier_x_norm` (geometry) |
| `d_P_total_dt` | `P_total` (pressure geometry) |
| `d_carrier_boundary_pressure_dt` | `carrier_boundary_pressure` (boundary geometry; source value is not itself one of the 19 inputs) |
| `d_vor_carrier_area_share` | `vor_carrier_area_share` (Voronoi geometry/network) |
| `d_vor_forward_receiver_area_share` | `vor_forward_receiver_area_share` (Voronoi geometry/network) |

Despite the `_dt` names, these are **current minus previous values, not
per-second rates**. The first anchor can use a valid pre-sequence 360 context;
otherwise its change is missing. The model handles missing feature values.
