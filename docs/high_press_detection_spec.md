---
name: high-press-detection
description: >
  Defines and implements the full pipeline for detecting high-press events and
  high-press sequences from StatsBomb 360 data. Use this skill whenever the
  user wants to identify high-pressing situations, compute off-ball pressure,
  build pressure networks, filter pressing sequences, or analyse high-pressure
  tactical events from StatsBomb event + freeze-frame data. Triggers on phrases
  like "high press", "pressing events", "pressure sequence", "off-ball pressure",
  "freeze frame pressure", "局部高压", "高位压迫".
---

# High-Press Detection — StatsBomb 360

## Overview

This skill defines a two-layer detection pipeline:

1. **Pressure calculation** — distance-based formulation for each freeze frame
2. **Event & sequence detection** — identifying high-press events and grouping
   them into continuous sequences

Read this file top to bottom before writing any code. All constants are
defined in §1 and must not be changed without explicit user instruction.

---

## §1  Parameter Reference

### 1.1 Pitch conventions (StatsBomb)

| Property | Value |
|---|---|
| x range | 0 – 120 (left → right, attacking direction) |
| y range | 0 – 80 |
| Centre line | x = 60 |
| Attacking direction | left → right (home team, 1st half) |
| Period correction | 2nd half: flip x → `120 - x` before comparison |

### 1.2 Player pressure parameters

| Symbol | Value | Meaning |
|---|---|---|
| D\*_player | 6.4 m | Effective interception radius |
| σ_d | 2.2 m | Logistic spread (distance domain) |
| k_player | π / (√3 × 2.2) ≈ **0.821** | Logistic steepness |

### 1.3 Boundary pressure parameters (sideline + byline)

Boundary params are exactly **half** of player params:

| Symbol | Value | Derivation |
|---|---|---|
| D\*_line | 3.2 m | D\*_player / 2 |
| σ_d_line | 1.1 m | σ_d / 2 |
| k_line | π / (√3 × 1.1) ≈ **1.643** | 2 × k_player |

### 1.4 High-press thresholds

| Parameter | Value | Notes |
|---|---|---|
| P_threshold | **0.65** | Minimum P_total for a high-press event |
| D\*_player | 6.4 m | At least 1 defender must be within this range |
| T_gap | **5 s** | Max time gap between consecutive events in a sequence |

---

## §2  Pressure Formulas

### 2.1 Unified logistic (same function, different params)

```
p = sigmoid(k, D*, d) = [1 + exp(-k · (D* - d))]^(-1)
```

### 2.2 Player pressure (defender i → ball carrier j)

```
D_i,j   = ||r_j - r_i||                         Euclidean distance [m]
p_i,j   = sigmoid(k_player, D*_player, D_i,j)
```

### 2.3 Boundary pressure (nearest line only)

```
d_line  = min(x_j, 120-x_j, y_j, 80-y_j)       nearest boundary [m]
p_bnd   = sigmoid(k_line, D*_line, d_line)
```

> **Corner exception**: if analysing corner / wide areas where two boundaries
> are both close, use all four lines:
> `p_bnd = 1 - ∏_b (1 - sigmoid(k_line, D*_line, d_b))`

### 2.4 Total pressure on ball carrier j

```
P_j = 1 - (1 - p_bnd) · ∏_i (1 - p_i,j)
```

Boundary enters the product identically to a single defender.
Independence between defenders is assumed (naive assumption, per Spearman).

---

## §3  High-Press Event Definition

A single event qualifies as a **high-press event** if ALL of the following hold:

| # | Condition | Implementation note |
|---|---|---|
| 1 | Event type ∈ {`pass`, `carry`, `miscontrol`, `dribble`} | `event['type']['name']` |
| 2 | Not a set piece | `event['play_pattern']['name']` not in {`From Free Kick`, `From Corner`, `From Throw In`, `From Kick Off`} |
| 3 | Ball carrier is in **their own half** | After period correction: `x_actor < 60` |
| 4 | P_total > 0.65 | Computed from freeze frame using §2 |
| 5 | ≥ 1 outfield defender within D\*_player = 6.4 m | Prevents boundary-only false positives |

### 3.1 Ball carrier identification

```python
# Ball carrier = actor in freeze frame
actor_pos = event['location']           # [x, y], from the event itself
defenders = [p['location'] for p in freeze_frame
             if not p['teammate']
             and not p.get('actor', False)
             and not p.get('keeper', False)]   # exclude GK
```

### 3.2 Period correction for attacking direction

```python
def to_attacking_frame(x, period, team_id, home_team_id):
    """
    StatsBomb: home team attacks left→right in period 1,
    right→left in period 2. Away team is reversed.
    Returns x normalised so that x < 60 = own half for the ball carrier.
    """
    attacks_left_to_right = (team_id == home_team_id) == (period == 1)
    return x if attacks_left_to_right else 120 - x
```

---

## §4  High-Press Sequence Definition

A **high-press sequence** is a maximal chain of consecutive high-press events
satisfying all four adjacency conditions:

| # | Condition |
|---|---|
| A | Both events are high-press events (P > 0.65, all criteria in §3) |
| B | Both events belong to the **same period** (`period` field) |
| C | Both events involve a ball carrier from the **same team** |
| D | Time gap between events ≤ T_gap = 5 s (`timestamp` difference) |

**Sequence boundaries**:
- **Start**: first event in chain (no valid predecessor)
- **End**: last event in chain (no valid successor)

### 4.1 Sequence metadata to extract

For each detected sequence, record:

```
sequence_id         unique identifier
start_event_id      StatsBomb event UUID
end_event_id        StatsBomb event UUID
team_id             team under pressure
period              match period
start_time          timestamp of first event
end_time            timestamp of last event
duration_s          end_time - start_time
n_events            number of events in sequence
mean_pressure       mean P_total across events
max_pressure        max P_total across events
outcome             type of last event (pass / miscontrol / carry / dribble)
```

---

## §5  Implementation Checklist

When implementing this pipeline:

1. [ ] Load StatsBomb events + 360 freeze frames, join on `event_uuid`
2. [ ] Apply period correction to all x-coordinates
3. [ ] Filter to event types in §3, condition 1–2
4. [ ] For each qualifying event, extract actor position + defender positions
5. [ ] Compute P_total using §2 formulas with parameters from §1
6. [ ] Apply conditions 3–5 to label each event as high-press or not
7. [ ] Sort events by `(period, timestamp)`
8. [ ] Apply adjacency conditions A–D to group sequences (§4)
9. [ ] Extract sequence metadata (§4.1)

---

## §6  Key Assumptions & Limitations

| Item | Assumption | Impact if violated |
|---|---|---|
| Velocities | All player velocities = 0 (360 data has no tracking) | Underestimates pressure from players running toward ball carrier |
| GK excluded | Goalkeeper not counted as defender | Slightly underestimates pressure near goal |
| Independence | Defender pressures are independent | Overestimates total pressure when defenders cluster |
| Nearest line only | Only min(4 boundary distances) used | Underestimates pressure in corners (use all-lines mode there) |
| T_gap = 5s | Fixed; not data-driven | May merge or split sequences incorrectly around dead balls |

---

## §7  Reference Code

Reference implementation is in `pressure_distance_v2.py`.

Key functions:

| Function | Purpose |
|---|---|
| `sigmoid_pressure(k, D_star, dist)` | Unified logistic — §2.1 |
| `player_pressure(ri, rj)` | Single defender pressure — §2.2 |
| `boundary_pressure(rj)` | Boundary pressure — §2.3 |
| `total_pressure(rj, defenders)` | P_total with breakdown — §2.4 |
| `analyze_freeze_frame(freeze_frame, ball_pos)` | Full freeze-frame pipeline |

Parameters live in the `PressureParams` dataclass. Do not hardcode constants;
always instantiate `PressureParams()` and read properties from it.

