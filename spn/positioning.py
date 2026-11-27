"""Compare a recorded presser's observed position with feasible nearby positions in one match."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spn._core.regain_labels import label_onset
from spn._core.carrier_identity import Timeline, infer, masks
from spn._core.position_sampler import candidates
from spn._core.named_pressure import actor_index, pressure_view
from spn._core.player_moves import move_named
from spn._core.network_features import DELTAS, shape_features, static_features
from spn._core.networks import build_network, team_shapes
from spn._core.positioning import physical_players, move_reason, strict_visible
from spn._core.pressure_events import Rules, pressure, team
from spn._core.receiver_features import receiver_features
from spn._core.pressure_context import onset_context
from spn._core.spatial_features import spatial_features
from .feature_engineering import INPUTS
from .model import MODEL_DIR, load_booster, probabilities, verified_path


def regain_inputs(event, frame, context):
    """Measure the 26 binary-model inputs at one position; preceding play stays fixed across moves."""
    network = build_network(event, frame)
    graphs, _ = team_shapes(network, corrected=True)
    row = static_features(network, event, frame)
    row.update(shape_features(network, graphs, corrected=True))
    row.update(
        {
            name: row[source] - context["previous"].get(source, np.nan)
            for name, source in DELTAS.items()
        }
    )
    row.update(context["constants"])
    row.update(receiver_features(network))
    row.update(spatial_features(network, graphs)[0])
    return {name: row[name] for name in INPUTS}


def pressure_positions(events, frames, match_id, *, move_players=True):
    """Yield each eligible recorded pressure and its ordered list of 26-feature vectors.

    A pressure in the broader regain cohort contributes its observed vector. For a
    high press, move_players also returns vectors for feasible alternatives in the
    same frame. A player's location changes in meters; every other player, the
    identity of the carrier, the event outcome and preceding play remain fixed.
    """
    teams = {team(e) for e in events if team(e) is not None}
    if len(teams) != 2:
        raise ValueError("Expected two teams in the event file")
    timeline = Timeline(events)
    rules = Rules()
    for index, event in enumerate(events):
        if event["type"]["name"] != "Pressure" or not team(event, "player"):
            continue
        uid = event["id"]
        frame = frames.get(uid)
        actor = actor_index(frame)
        if actor is None:
            continue
        player = frame["freeze_frame"][actor]
        if (
            player.get("keeper")
            or not event.get("location")
            or np.linalg.norm((np.asarray(player["location"]) - event["location"]) * [0.875, 0.85])
            > 1e-7
        ):
            continue
        pressing = team(event)
        (owner,) = teams - {pressing}
        outcome = label_onset(events, index, pressing, owner)["e2"]
        if outcome not in (0, 1):
            continue
        # A Pressure frame identifies the presser. Locate the carrier using control
        # evidence within half a second and a unique correspondence between frames.
        ref = infer(
            event,
            frame,
            timeline.evidence(event),
            frames,
            owner,
            0.5,
            "partial_fallback",
            {},
        )
        if ref["reason"] != "accepted" or len(frame["freeze_frame"]) < 4:
            continue
        carrier = ref["carrier_index"]
        view, oriented = pressure_view(event, frame, owner, carrier, {"id": ref["carrier_id"]})
        try:
            visible = strict_visible(oriented, False)
        except ValueError as error:
            if str(error) in (
                "missing_or_nonfinite_visible_polygon",
                "invalid_visible_polygon",
            ):
                continue
            raise
        # Eligibility requires four feasible directions on each of the 1.25 m and
        # 2 m rings. The 64 area-uniform alternatives below are a separate sample.
        rings = masks(oriented, actor, carrier)
        if any(bin(mask).count("1") < 4 for mask in rings.values()):
            continue
        context, _ = onset_context(events, event, view, frames)
        measured = pressure(view["location"], oriented, rules)
        high = bool(
            view["location"][0] < rules.own_half_x
            and measured
            and measured["n_in_D"] >= 1
            and measured["P_total"] > rules.pressure_threshold
        )
        _, xy = physical_players(oriented, False)
        # Preserve the meter-to-provider-grid round trip used for the observed position in the fits.
        original, _ = move_named(oriented, actor, xy[actor])
        vectors = [regain_inputs(view, original, context)]
        if high and move_players:
            sampled = candidates(str(match_id), uid, xy[actor], xy[carrier])
            for target in sampled[["target_x", "target_y"]].to_numpy():
                if move_reason(target, visible, xy, actor) == "valid":
                    moved, _ = move_named(oriented, actor, target)
                    vectors.append(regain_inputs(view, moved, context))
        yield (
            {
                "match_id": str(match_id),
                "event_id": uid,
                "player_id": team(event, "player"),
                "pressing_team_id": pressing,
                "high": high,
                "e2": int(outcome),
            },
            vectors,
        )


def position_quality(observed, alternatives):
    """Return observed minus mean alternative regain probability, in percentage points."""
    values = np.asarray(alternatives, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or not np.all((values >= 0) & (values <= 1))
        or not 0 <= observed <= 1
    ):
        raise ValueError("Expected one observed probability and at least one valid alternative")
    return 100 * (float(observed) - values.mean())


class PositioningModel:
    """Use the fixture's held-out regain model, or all five folds on a new match."""

    def __init__(self, config_path=None):
        config_path = Path(config_path or MODEL_DIR / "model_config.json")
        config = json.loads(config_path.read_text())
        self.model_id = config["model_id"]
        self.directory = config_path.parent
        self.checksums = config["artifact_checksums"]
        settings = config["positioning"]
        self.artifacts = settings["fold_artifacts"]
        fold_file = settings["fold_map"]
        fold_path = verified_path(self.directory / fold_file, self.checksums[fold_file])
        self.fold_map = json.loads(fold_path.read_text())
        self.models = {}

    def predict(self, match):
        match_id = str(match.match_id)
        folds = (
            [self.fold_map[match_id]] if match_id in self.fold_map else range(len(self.artifacts))
        )
        models = []
        for fold in folds:
            if fold not in self.models:
                name = self.artifacts[fold]
                self.models[fold] = load_booster(
                    verified_path(self.directory / name, self.checksums[name])
                )
            models.append(self.models[fold])
        rows = []
        for info, vectors in pressure_positions(match.events, match.frames, match.match_id):
            if not info["high"] or len(vectors) < 2:
                continue
            table = pd.DataFrame(vectors, columns=INPUTS)
            p = np.mean([probabilities(model, table).astype(float) for model in models], axis=0)
            rows.append(
                dict(
                    info,
                    valid_alternatives=len(p) - 1,
                    observed_probability=p[0],
                    alternative_probability=p[1:].mean(),
                    position_quality_pp=position_quality(p[0], p[1:]),
                )
            )
        return pd.DataFrame(
            rows,
            columns=[
                "match_id",
                "event_id",
                "player_id",
                "pressing_team_id",
                "high",
                "e2",
                "valid_alternatives",
                "observed_probability",
                "alternative_probability",
                "position_quality_pp",
            ],
        )


def score_positions(match, model=None):
    """Score eligible recorded pressers in one MatchData object, in percentage points."""
    return (model or PositioningModel()).predict(match)
