# The 26 model inputs

`spn.feature_engineering.INPUTS` gives the column order stored in both models. The first eight entries and entry 21 form the nine-input baseline in Figure 2c. All 26 enter the full model. Visible players define the passing, pressure and space calculations. Distances are in meters after rescaling to a 105 × 68 m pitch. Forward is the attacking direction of the team in possession.

| Column | Interpretation |
| --- | --- |
| `ball_in_dist` | Distance traveled from the preceding usable ball location to the current location. |
| `ball_in_angle_cos` | Cosine of the incoming ball's direction relative to the attacking axis. |
| `carrier_x_norm` | Carrier's longitudinal location as a fraction of pitch length. |
| `d_carrier_x_norm_dt` | Change in that fraction since the preceding represented position. |
| `carrier_touchline_m` | Distance to the nearer touchline. |
| `P_total` | Combined direct pressure on the carrier, calculated as one minus the product of the individual pressure complements. |
| `d_P_total_dt` | Change in combined carrier pressure. |
| `effective_pressers` | Effective number of pressers on the carrier, the squared sum of their retained pressure weights divided by the sum of squared weights. |
| `n_open_pass` | Number of visible passing options with a passing weight above 0.5. |
| `best_forward_pass_w` | Largest passing weight among forward receivers. |
| `press_target_entropy` | Normalized entropy of pressure across the carrier and receivers. Higher values indicate more even coverage. |
| `mean_receiver_pressure` | Mean across receivers of the summed retained pressure on each. |
| `pressure_mean_focus_entropy` | Mean across active pressers of the normalized entropy of each presser's pressure across targets. |
| `vor_carrier_area_share` | Carrier's Voronoi cell area divided by visible pitch area. |
| `vor_forward_receiver_area_share` | Sum of forward receivers' cell areas divided by visible pitch area. |
| `d_vor_carrier_area_share` | Change in the carrier's share of visible space. |
| `d_vor_forward_receiver_area_share` | Change in the forward receivers' share of visible space. |
| `sg_n_outlets_uncrossed` | Number of carrier-to-outfield-teammate segments that cross no pressing-team shape edge. |
| `sg_pressed_edges_cut_frac_2m` | Fraction of the in-possession team's shape edges with a presser less than 2 m away whose projection falls on the edge. |
| `sg_press_max_edge` | Longest pressing-team shape edge, in meters. |
| `seq_after_opponent` | Indicator that the preceding context event selected for the sequence was by the opposing team. |
| `best_forward_receiver_relief` | Largest product of passing weight and receiver freedom among forward receivers. Freedom is the product of one minus each defender's pressure on that receiver. |
| `forward_pass_space_share` | Sum across forward receivers of each passing weight times that receiver's share of the visible area. |
| `local_defender_spread_m` | Mean distance of the five pressers nearest the carrier from their common centroid. |
| `nearest_presser_support_m` | Median length of the pressing team's shape edges incident to the nearest presser. |
| `carrier_forward_space_fraction` | Fraction of the carrier's visible Voronoi cell lying forward of the carrier. |

Changes compare successive represented positions; the stored column names ending in `_dt` denote differences per step. At onset the code uses eligible preceding context where available. Missing context or geometry can produce a missing value, which XGBoost handles natively. A valid frame can also yield a genuine zero, for example when there is no forward pass. Preserve that distinction when scoring new data.

For exact reproduction, the sequence features use the shape-graph implementation used for the archived sequence fit. The positioning analysis uses the teamwise implementation to calculate each side's shape for the observed frame and each feasible alternative. The two functions are in [vendor/](vendor/). The sequence and positioning entry points select the implementation used by their respective fits.
