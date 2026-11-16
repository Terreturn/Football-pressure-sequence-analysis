# SPN Classifier

SPN Classifier, version `1.0.0`, is a sequence-weighted XGBoost classifier with
the `multi:softprob` objective and three outcome classes. The trained trees are
stored in `model.ubj`. Each training sequence contributes total weight 1,
distributed equally across its anchors as `1 / anchor_count`.

The model takes 19 numeric SPN features for each high-pressure sequence anchor,
computed from paired StatsBomb event and 360 data. Feature names and their input
order are defined in the configuration. The prediction interface also requires
`match_id`, `seq_id`, `ev_pos`, and `anchor_event_id` to identify each row.
[Model configuration](model_config.json)

Each anchor receives raw probabilities `p_fail`, `p_neutral`, and `p_success`
for the sequence outcome. The output retains the four row identifiers and adds
`predicted_class`, the class with the highest probability, and `confidence`,
that probability value.

The saved model was trained with the following settings.

| Parameter | Value |
| --- | --- |
| `n_estimators` (boosting rounds) | 500 |
| `max_depth` | 6 |
| `learning_rate` | 0.03 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 5 |
| `gamma` | 1 |
| `reg_lambda` | 3 |

Preprocessing uses a player-pressure distance centre of 6.4 m, a boundary-pressure
centre of 3.2 m, and a passing-lane centre of 1.2 m. Sequence detection uses a
pressure threshold of 0.65, an own-half limit of `x < 60` in StatsBomb coordinates,
and a maximum anchor gap of 5 s. Full preprocessing settings, class order, and
the model file checksum are recorded in the configuration.
[Model configuration](model_config.json)

Feature definitions and further model details are provided in the existing
model documentation. [Model and selected features](../spn/MODEL.md)
