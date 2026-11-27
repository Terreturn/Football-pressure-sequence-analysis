"""Structural Pressure Network match analysis.

data_processing loads matches and labels pressure sequences; feature_engineering
measures the 26 inputs. model scores sequence outcomes and positioning scores
individual pressers. output builds the result tables and plotting draws them.
"""

from .data_processing import MatchData, load_match_data, build_sequences_from_match
from .feature_engineering import build_feature_table
from .model import SPNModel
from .positioning import PositioningModel, score_positions
from .output import build_result, save_result

__all__ = [
    "MatchData",
    "SPNModel",
    "PositioningModel",
    "score_positions",
    "build_feature_table",
    "build_result",
    "build_sequences_from_match",
    "load_match_data",
    "save_result",
]
