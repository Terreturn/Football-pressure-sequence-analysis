"""Public network parameters and construction on the 105 × 68 m pitch."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ._core.networks import Parameters, build_network

P_EPS = 0.05


@dataclass(frozen=True)
class PressureParams:
    player_distance: float = 6.4
    player_sd: float = 2.2
    boundary_distance: float = 3.2
    boundary_sd: float = 1.1
    lane_distance: float = 1.2
    lane_sd: float = 0.6
    pass_decay_m: float = 18.0
    pass_open_threshold: float = 0.5
    edge_epsilon: float = P_EPS

    def __post_init__(self) -> None:
        positive = {
            "player_distance": self.player_distance,
            "player_sd": self.player_sd,
            "boundary_distance": self.boundary_distance,
            "boundary_sd": self.boundary_sd,
            "lane_distance": self.lane_distance,
            "lane_sd": self.lane_sd,
            "pass_decay_m": self.pass_decay_m,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"pressure/network parameters must be positive: {invalid}")
        if not 0.0 < self.pass_open_threshold < 1.0:
            raise ValueError("pass_open_threshold must be between zero and one")
        if not 0.0 < self.edge_epsilon < 1.0:
            raise ValueError("edge_epsilon must be between zero and one")

    @property
    def k_player(self) -> float:
        return math.pi / (math.sqrt(3.0) * self.player_sd)

    @property
    def k_boundary(self) -> float:
        return math.pi / (math.sqrt(3.0) * self.boundary_sd)

    @property
    def k_lane(self) -> float:
        return math.pi / (math.sqrt(3.0) * self.lane_sd)


def network_parameters(params: PressureParams) -> Parameters:
    return Parameters(
        player_mid=params.player_distance,
        player_sd=params.player_sd,
        line_mid=params.boundary_distance,
        line_sd=params.boundary_sd,
        lane_mid=params.lane_distance,
        lane_sd=params.lane_sd,
        decay=params.pass_decay_m,
        epsilon=params.edge_epsilon,
        open_threshold=params.pass_open_threshold,
    )


def build_spn_network(event: dict, frame: dict, params: PressureParams = PressureParams()) -> dict:
    return build_network(event, frame, network_parameters(params))
