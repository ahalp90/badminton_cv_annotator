"""rtmlib pose-extraction adapter.

Drives rtmlib's low-level RTMDet (person detector) + RTMPose (COCO-17 estimator)
over onnxruntime to reproduce the per-person output the mmpose extraction path
consumed -- ``keypoints`` / ``bbox`` / ``bbox_score`` / ``keypoint_scores`` --
without the mmcv / mmdet / mmpose / mmengine stack or its ``numpy < 2`` pin. It
replaces ``MMPoseInferencer("human")`` for ``raw_extract`` and
``detect_players_2d``.

The default models mirror ``MMPoseInferencer("human")``: the ``rtmdet-nano``
person ONNX is byte-identical to mmpose's detector, and ``rtmpose-l body7`` is
the updated (7-dataset) RTMPose-L, COCO-17.

Two rtmlib quirks are corrected here:

* ``RTMDet.postprocess`` computes the per-box detection score then returns only
  the boxes; ``RTMDetScored`` recovers the score column (``raw_extract`` and the
  ``sticky_anchor`` heuristic both need ``bbox_score``).
* ``RTMPose`` normalises with an RGB-order mean but never converts BGR->RGB
  (an undocumented rtmlib bug). We feed it an RGB crop so the colour order
  matches the trained model; the detector keeps its native BGR input.

The N_max cap, NaN padding and dtype are the *caller's* concern (``raw_extract``
pads to N_max as float32; ``detect_players_2d`` casts to float64 to match the old
``np.array(list)`` path). This module returns only the real detections.

Consumers import this module lazily (inside their functions) so
``prepare_train_on_shuttleset`` stays importable without onnxruntime installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, NamedTuple

import cv2
import numpy as np
from rtmlib.tools.object_detection.rtmdet import RTMDet
from rtmlib.tools.pose_estimation.rtmpose import RTMPose

J = 17  # COCO keypoints (RTMPose-L body7), matching the mmpose extract.

# rtmlib-loadable mmdeploy ONNX-SDK archives. Detector hash 05d8511e is identical
# to the one inside MMPoseInferencer("human"); the pose model is the updated
# body7 RTMPose-L. Pin + SHA-verify via validation_scripts/rtmlib_migration.
_MODEL_BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
DET_URL = _MODEL_BASE + "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.zip"
POSE_URL = _MODEL_BASE + "rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.zip"
DET_INPUT_SIZE = (320, 320)   # fixed by the person ONNX export
POSE_INPUT_SIZE = (192, 256)  # (W, H), i.e. 256x192
DET_SCORE_THR = 0.3           # mmpose's detector kept score > ~0.30 (min observed 0.301)


class FrameDetections(NamedTuple):
    """The ``m`` real (unpadded) detections in one frame, detector slot order.

    NaN-padding and the N_max cap are a ``raw_extract`` concern; this carries
    only the people actually found (``m`` may be 0).
    """
    keypoints: np.ndarray    # (m, J, 2) float32; image-pixel coords, COCO-17 order
    bboxes: np.ndarray       # (m, 4) float32; xyxy image pixels
    bbox_scores: np.ndarray  # (m,) float32; per-person detection confidence
    kp_scores: np.ndarray    # (m, J) float32; per-joint confidence


class RTMDetScored(RTMDet):
    """``RTMDet`` that also returns the per-box detection score.

    The person ONNX has NMS baked in (output ``dets`` shape ``(1, N, 5)``), so
    the score is column 4; stock rtmlib discards it. Detection score is required
    downstream (``_raw_scores`` -> ``sticky_anchor``), so we return it alongside
    the boxes and filter on ``self.score_thr``.
    """

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image, ratio = self.preprocess(image)
        outputs = self.inference(image)[0]
        if outputs.shape[-1] != 5:
            raise RuntimeError(
                f"expected an NMS-baked detector output (..., 5); got {outputs.shape}. "
                "The default rtmdet-nano-person ONNX bakes in NMS; a different "
                "detector needs the grid-decode branch handled too."
            )
        boxes = (outputs[0, :, :4] / ratio).astype(np.float32)  # (N, 4) xyxy, orig pixels
        scores = outputs[0, :, 4].astype(np.float32)            # (N,)
        keep = scores > self.score_thr
        return boxes[keep], scores[keep]


class RtmlibPoseExtractor:
    """Loads the detector + pose models once and extracts per-frame detections.

    CPU inference is deterministic run-to-run at a fixed thread count. rtmlib's
    ``BaseTool`` builds the onnxruntime session with no ``SessionOptions``, so
    ``intra_op_num_threads`` is not a constructor knob -- pin threads via
    ``OMP_NUM_THREADS`` in the environment when a caller needs bit-reproducibility
    (the CPU determinism gate does). CUDA is nondeterministic regardless.

    :param device: onnxruntime device -- ``"cpu"`` or ``"cuda"`` (needs
        ``onnxruntime-gpu`` for the latter).
    :param det_url: person-detector ONNX (defaults to the mmpose-identical rtmdet-nano).
    :param pose_url: pose ONNX (defaults to RTMPose-L body7 COCO-17, 256x192).
    :param det_input_size: detector input (H, W); fixed at 320x320 for the default ONNX.
    :param pose_input_size: pose input (W, H); 192x256 for the default 256x192 model.
    :param det_score_thr: keep detections with score above this (0.30 mirrors mmpose).
    """

    def __init__(
        self,
        device: str = "cpu",
        *,
        det_url: str = DET_URL,
        pose_url: str = POSE_URL,
        det_input_size: tuple[int, int] = DET_INPUT_SIZE,
        pose_input_size: tuple[int, int] = POSE_INPUT_SIZE,
        det_score_thr: float = DET_SCORE_THR,
    ) -> None:
        self.det = RTMDetScored(det_url, model_input_size=det_input_size, device=device)
        self.det.score_thr = det_score_thr
        self.pose = RTMPose(pose_url, model_input_size=pose_input_size, device=device)

    def detect_frame(self, frame_bgr: np.ndarray) -> FrameDetections:
        """Run detector + pose on one BGR frame; return the ``m`` real detections.

        On zero detections we return empty ``(m=0)`` arrays rather than let
        RTMPose fall back to a whole-image "person" (its stock empty-bbox
        behaviour, which would fabricate a spurious detection).

        :param frame_bgr: (H, W, 3) uint8 BGR image (OpenCV convention).
        :return: ``FrameDetections`` for the ``m`` people found.
        """
        boxes, bbox_scores = self.det(frame_bgr)  # (m, 4), (m,)
        if len(boxes) == 0:
            return FrameDetections(
                keypoints=np.empty((0, J, 2), dtype=np.float32),
                bboxes=np.empty((0, 4), dtype=np.float32),
                bbox_scores=np.empty((0,), dtype=np.float32),
                kp_scores=np.empty((0, J), dtype=np.float32),
            )
        # RGB fix: RTMPose's mean is RGB-order but it never converts, so hand it RGB.
        # warpAffine is colour-agnostic, so convert-then-warp == mmpose's warp-then-convert.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        keypoints, kp_scores = self.pose(frame_rgb, bboxes=boxes)  # (m, J, 2), (m, J)
        return FrameDetections(
            keypoints=keypoints.astype(np.float32),
            bboxes=boxes,
            bbox_scores=bbox_scores,
            kp_scores=kp_scores.astype(np.float32),
        )

    def iter_video(self, video_path: Path | str) -> Iterator[FrameDetections]:
        """Yield ``FrameDetections`` for each frame of an mp4, in decode order.

        Uses ``cv2.VideoCapture``, whose per-clip frame count was validated to
        match mmpose's decoder (``dF = 0`` across the parity sample).

        :param video_path: path to the clip .mp4.
        :return: generator of ``FrameDetections``, one per decoded frame.
        """
        cap = cv2.VideoCapture(str(video_path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield self.detect_frame(frame)
        finally:
            cap.release()
