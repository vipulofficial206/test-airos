"""
AirOS++ Hand Tracker
Provides temporal ID association, velocity estimation, and graceful missing detection extrapolation.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from airos.config.settings import DetectorConfig
from airos.detector.hand_detector import HandDetection
from airos.logger.airos_logger import get_logger

logger = get_logger()


@dataclass
class TrackedHand:
    """Dataclass storing temporal tracking state for an individual hand."""

    track_id: int
    label: str  # 'left' or 'right'
    history: List[HandDetection] = field(default_factory=list)
    last_seen_time: float = field(default_factory=time.time)
    consecutive_frames: int = 1
    missing_frames: int = 0
    predicted_centroid: Optional[Tuple[float, float]] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    baseline_aspect_ratio: float = 1.0


class HandTracker:
    """Temporal tracker matching current detections to existing hand trajectories."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.tracks: Dict[int, TrackedHand] = {}
        self.next_track_id: int = 1
        self.max_history: int = config.temporal_memory_frames

    def update(self, detections: List[HandDetection]) -> List[HandDetection]:
        """Matches new detections to active tracks via minimum centroid distance association."""
        current_time = time.time()
        updated_detections: List[HandDetection] = []

        # Predict next centroid positions for active tracks based on last velocity
        for track_id, track in self.tracks.items():
            if track.history:
                last_det = track.history[-1]
                vx, vy = track.velocity
                dt = max(0.001, current_time - track.last_seen_time)
                # Extrapolate position if missing
                px = last_det.centroid[0] + vx * dt
                py = last_det.centroid[1] + vy * dt
                track.predicted_centroid = (px, py)

        # Match detections to existing tracks using greedy spatial distance
        unmatched_detections = list(detections)
        matched_track_ids = set()

        for det in list(unmatched_detections):
            best_track_id: Optional[int] = None
            min_dist = float("inf")

            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue
                # Compare label preference or spatial distance
                ref_pt = (
                    track.predicted_centroid
                    if track.predicted_centroid
                    else track.history[-1].centroid
                )
                dist = math.hypot(det.centroid[0] - ref_pt[0], det.centroid[1] - ref_pt[1])
                if dist < 120.0 and dist < min_dist:  # Max association distance
                    min_dist = dist
                    best_track_id = track_id

            if best_track_id is not None:
                track = self.tracks[best_track_id]
                matched_track_ids.add(best_track_id)
                unmatched_detections.remove(det)

                # Compute velocity vector
                prev_det = track.history[-1]
                dt = max(0.001, det.timestamp - prev_det.timestamp)
                vx = (det.centroid[0] - prev_det.centroid[0]) / dt
                vy = (det.centroid[1] - prev_det.centroid[1]) / dt
                track.velocity = (vx, vy)

                # Update track history
                track.history.append(det)
                if len(track.history) > self.max_history:
                    track.history.pop(0)

                track.last_seen_time = det.timestamp
                track.consecutive_frames += 1
                track.missing_frames = 0
                track.predicted_centroid = None

                # Keep baseline aspect ratio updated via running mean during stable periods
                if len(track.history) >= 3:
                    ar_vals = [d.aspect_ratio for d in track.history[-5:]]
                    track.baseline_aspect_ratio = float(np.median(ar_vals))

                det.track_id = best_track_id
                det.label = track.label  # Maintain spatial identity
                updated_detections.append(det)

        # Handle unmatched detections -> Create new tracks
        for det in unmatched_detections:
            new_id = self.next_track_id
            self.next_track_id += 1
            label = det.label if det.label != "unassigned" else "left"

            track = TrackedHand(
                track_id=new_id,
                label=label,
                history=[det],
                last_seen_time=det.timestamp,
                consecutive_frames=1,
                baseline_aspect_ratio=det.aspect_ratio,
            )
            self.tracks[new_id] = track
            det.track_id = new_id
            updated_detections.append(det)

        # Handle missing tracks (hand temporary disappearance)
        dead_track_ids = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_track_ids:
                track.missing_frames += 1
                track.consecutive_frames = 0
                if track.missing_frames > self.config.extrapolation_max_frames:
                    dead_track_ids.append(track_id)

        for tid in dead_track_ids:
            del self.tracks[tid]

        return updated_detections

    def get_track(self, track_id: int) -> Optional[TrackedHand]:
        return self.tracks.get(track_id)
