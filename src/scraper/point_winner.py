"""Old-path import shim; module moved to annotator.point_winner (Stage 2).

Import-only (no -m surface); dies at Stage 7.
"""
from annotator.point_winner import Half as Half
from annotator.point_winner import HitHeightRow as HitHeightRow
from annotator.point_winner import Landing as Landing
from annotator.point_winner import LandingFilterOptions as LandingFilterOptions
from annotator.point_winner import LandingKinematics as LandingKinematics
from annotator.point_winner import Verdict as Verdict
from annotator.point_winner import VerdictSource as VerdictSource
from annotator.point_winner import OTHER_HALF as OTHER_HALF
from annotator.point_winner import HOMOGRAPHY_RESOLUTION as HOMOGRAPHY_RESOLUTION
from annotator.point_winner import NET_COURT_Y as NET_COURT_Y
from annotator.point_winner import attribute_half as attribute_half
from annotator.point_winner import build_hit_height_rows as build_hit_height_rows
from annotator.point_winner import build_landing_kinematics as build_landing_kinematics
from annotator.point_winner import corner_error_band_m as corner_error_band_m
from annotator.point_winner import court_scale_boxes as court_scale_boxes
from annotator.point_winner import convert_landing_options as convert_landing_options
from annotator.point_winner import filtered_descending_landing as filtered_descending_landing
from annotator.point_winner import fit_alternation as fit_alternation
from annotator.point_winner import geometric_verdict as geometric_verdict
from annotator.point_winner import hit_height as hit_height
from annotator.point_winner import inout_verdict as inout_verdict
from annotator.point_winner import is_net_ender as is_net_ender
from annotator.point_winner import landing_margins as landing_margins
from annotator.point_winner import next_server_half as next_server_half
from annotator.point_winner import pick_landing as pick_landing
from annotator.point_winner import project_pixels_to_court as project_pixels_to_court
from annotator.point_winner import rally_verdict as rally_verdict
from annotator.point_winner import window_end as window_end
from annotator.point_winner import _body_unit_gaps as _body_unit_gaps
from annotator.point_winner import _carried_terminal as _carried_terminal
