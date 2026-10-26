from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np
from uniaction.evaluation import plant_flow_inverse as pi
from uniaction.evaluation import plant_flow_inverse_regression as pr
from uniaction.evaluation import plant_flow_observability as po


class FlowerDetector:
    def __init__(self, descriptor_path: str | Path):
        self.descriptor_path = Path(descriptor_path)
        self.descriptor = json.loads(self.descriptor_path.read_text())
        metadata = self.descriptor
        if (
            metadata["schema"] != "uniaction-flower-detector"
            or metadata["config"]["method"] != "ridge"
        ):
            raise ValueError("unsupported flower calibration bundle")
        if cv2.__version__ != metadata["opencv_version"] or not hasattr(
            cv2, "DISOpticalFlow_create"
        ):
            raise ValueError("Flower scoring requires OpenCV 4.13.0 with DIS optical flow")

        def artifact(name):
            root = self.descriptor_path.resolve().parent
            path = (root / name).resolve()
            path.relative_to(root)
            return path

        with np.load(artifact(metadata["weights_path"]), allow_pickle=False) as weights:
            self.scales = weights["scales"].copy()
            self.coefficients = weights["coefficients"].copy()
            groups = weights["group_ids"]
        self.feature_config = pi.PlantFlowConfig(**metadata["feature_config"])
        # Regression weights are fitted on training data.
        self.config = metadata["config"]
        dimension = self.config["intervals"] * 5 * len(self.feature_config.radii_px) * 2
        if (
            not np.array_equal(groups, np.arange(25))
            or self.scales.shape != (25, dimension)
            or self.coefficients.shape != (25, dimension, 2)
        ):
            raise ValueError("calibration coefficient inventory mismatch")
        if (
            not np.isfinite(self.scales).all()
            or not np.isfinite(self.coefficients).all()
            or np.any(self.scales <= 0)
        ):
            raise ValueError("invalid calibration coefficients")
        with np.load(artifact(metadata["view_bank_path"]), allow_pickle=False) as bank:
            self.view_bank = pi.PlantViewBank(
                bank["reference_rgb"], bank["anchors_xy"], config=self.feature_config
            )

        self.observability_config = po.PlantObservabilityConfig(**metadata["observability_config"])

    def infer_features(self, features):
        if features["schema"] != "uniaction-flower-flow-features":
            raise ValueError("unsupported cached flow schema")
        if json.dumps(features["config"], sort_keys=True) != json.dumps(
            asdict(self.feature_config), sort_keys=True
        ):
            raise ValueError("cached flow configuration mismatch")
        if not features["valid"]:
            return {"valid": False, "reason": features["reason"], "direction_xy": None}
        group = features["registration"]["group_id"]
        if (
            group != 5 * features["registration"]["view_id"] + features["registration"]["point_id"]
            or not 0 <= group < 25
        ):
            raise ValueError("invalid registered camera/point group")
        query, mask = pi.feature_vector(
            features,
            intervals=self.config["intervals"],
            representation=self.config["representation"],
        )
        return pr.predict_ridge_direction(
            {"scale": self.scales[group], "coefficients": self.coefficients[group]}, query, mask
        )

    def _score_direction(self, features, requested_direction):
        predicted = self.infer_features(features)
        if requested_direction is None:
            return {
                "schema": "uniaction-flower-inferred-force",
                "score": 0.0,
                "applicable": False,
                "reason": "zero_length_condition",
                "direction_cosine": None,
                "prediction": predicted,
                "requested_direction_available": False,
            }
        transform = (
            np.asarray(features["registration"]["affine"])[:, :2]
            if features["valid"]
            else np.eye(2)
        )
        measured = pi.score_inferred_force(
            predicted,
            requested_direction=np.asarray(requested_direction),
            canonical_to_image=transform,
            motion_signal_px=features.get("motion_signal_px", 0),
            motion_threshold_px=self.config["motion_signal_threshold_px"],
        )
        return {**measured, "prediction": predicted, "requested_direction_available": True}

    def observe(self, frames, features):
        if not features["valid"]:
            return {
                "schema": "uniaction-plant-flow-observability-v1",
                "config": asdict(self.observability_config),
                "observable": False,
                "reason": features["reason"],
                "signal_px": 0.0,
                "traces": [],
            }
        reg = features["registration"]
        affine = np.asarray(reg["affine"])
        anchors = self.view_bank.anchors[reg["view_id"]] @ affine[:, :2].T + affine[:, 2]
        return po.check_plant_observability(
            frames, anchors_xy=anchors, config=self.observability_config
        )

    def score_features(self, features, observation, requested_direction):
        if observation["schema"] != "uniaction-plant-flow-observability-v1":
            raise ValueError("unsupported motion-observability schema")
        if json.dumps(observation["config"], sort_keys=True) != json.dumps(
            asdict(self.observability_config), sort_keys=True
        ):
            raise ValueError("Motion configuration does not match the detector")
        response = self._score_direction(features, requested_direction)
        if requested_direction is not None and (not observation["observable"]):
            response = {
                **response,
                "score": 0.0,
                "applicable": False,
                "reason": observation["reason"],
            }
        return {
            **response,
            "observability": observation,
            "schema": "uniaction-observable-flower-response",
        }

    def score_frames(self, frames, *, reference_rgb, start_xy, requested_direction):
        if np.asarray(frames).shape != (81, 480, 832, 3):
            raise ValueError("Expected 81 RGB frames at 832 x 480")
        features = pi.extract_plant_flow_features(
            frames,
            reference_rgb=reference_rgb,
            start_xy=start_xy,
            view_bank=self.view_bank,
        )
        observation = self.observe(frames, features)
        response = self.score_features(features, observation, requested_direction)
        return {
            "response": response,
            "registration": features["registration"],
        }
