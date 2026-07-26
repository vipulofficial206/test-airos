"""Detection subpackage for hand bounding box extraction and spatial feature computations."""
from airos.detector.hand_detector import HandDetection, HandDetector
from airos.detector.spatial_features import SpatialFeatureExtractor, SpatialFeatures

__all__ = ["HandDetection", "HandDetector", "SpatialFeatureExtractor", "SpatialFeatures"]
