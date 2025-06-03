from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from posenet.dataset.types import Batch
from posenet.misc.disk_cache import make_cache
from posenet.misc.nn_module_tools import convert_to_buffer
from .track_predictor import TrackPredictor, Tracks
from .track_predictor_cotracker import (
    TrackPredictorCoTracker,
    TrackPredictorCoTrackerCfg,
)

TRACKERS = {
    "cotracker": TrackPredictorCoTracker,
}

TrackPredictorCfg = TrackPredictorCoTrackerCfg


def get_track_predictor(cfg: TrackPredictorCfg) -> TrackPredictor:
    tracker = TRACKERS[cfg.name](cfg)
    convert_to_buffer(tracker, persistent=False)
    return tracker


def get_cache_key(
        dataset: str,
        indices: Int64[Tensor, " frame"],
        min_num_tracks: int,
        max_track_interval: int,
) -> tuple[str, int, int, int, int]:
    first, *_, last = indices
    return (
        dataset,
        first.item(),
        last.item(),
        min_num_tracks,
        max_track_interval,
    )


def generate_video_tracks(
        tracker: TrackPredictor,
        videos: Float[Tensor, "batch frame 3 height width"],
        interval: int,
        radius: int,
) -> list[Tracks]:
    segment_tracks = []
    b, f, _, _, _ = videos.shape
    start_end = []
    for middle_frame in range(0, f, interval):
        # Retrieve the video segment we want to compute tracks for.
            start_frame = max(0, middle_frame - radius)
            end_frame = min(f, middle_frame + radius + 1)
            if (start_frame, end_frame) not in start_end:
                start_end.append((start_frame, end_frame))
                tracks = None
                for cam in range(b):
                    segment = videos[cam, start_frame:end_frame][None]
                    # Compute tracks on the segment, then mark the tracks with the segment's
                    # starting frame so that they can be matched to the segment.
                    track = tracker.forward(segment, middle_frame - start_frame)

                    if tracks is None:
                        tracks = track
                        tracks.start_frame = start_frame
                    else:
                        tracks.xy = torch.cat([tracks.xy, track.xy], dim=0)
                        tracks.visibility = torch.cat([tracks.visibility, tracks.visibility], dim=0)
                segment_tracks.append(tracks)
            else:
                pass

    return segment_tracks


@dataclass
class TrackPrecomputationCfg:
    cache_path: Path | None
    interval: int
    radius: int


def compute_tracks(
        batch: Batch,
        device: torch.device,
        tracking_cfg: TrackPredictorCfg,
        precomputation_cfg: TrackPrecomputationCfg,
) -> list[Tracks]:
    # Set up the tracker.
    tracker = get_track_predictor(tracking_cfg)
    tracker.to(device)

    cache_key = get_cache_key(
        batch.datasets[0],
        batch.indices[0],
        precomputation_cfg.interval,
        precomputation_cfg.radius)
    disk_cache = make_cache(precomputation_cfg.cache_path)
    track = disk_cache(
        cache_key,
        lambda: generate_video_tracks(
            tracker,
            batch.videos.to(device),
            precomputation_cfg.interval,
            precomputation_cfg.radius,
        ),
    )
    return track


def compute_tracks_with_predictor(
        tracker: TrackPredictor,
        batch: Tensor,
        interval: int,
        radius: int,
        indices: list
) -> list[Tracks]:
    cache_key = get_cache_key(
        'l',
        indices,
        interval,
        radius)
    disk_cache = make_cache(None)
    track = disk_cache(
        cache_key,
        lambda: generate_video_tracks(
            tracker,
            batch,
            interval,
            radius,
        ),
    )

    # track = generate_video_tracks(
    #         tracker,
    #         batch,
    #         interval,
    #         radius,)
    return track
