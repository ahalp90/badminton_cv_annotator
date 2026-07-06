"""rtmlib pose-extraction adapter.

Drives rtmlib's low-level RTMDet (person detector) + RTMPose (COCO-17 estimator)
over onnxruntime to reproduce the per-person output the mmpose extraction path
consumed (``keypoints`` / ``bbox`` / ``bbox_score`` / ``keypoint_scores``)
without the mmcv / mmdet / mmpose / mmengine stack or its ``numpy < 2`` pin. It
replaces ``MMPoseInferencer("human")`` for ``raw_extract`` and
``detect_players_2d``.

The default detector is the ONNX export of the same RTMDet-M person checkpoint
(235e8209) that ``MMPoseInferencer("human")`` resolves at mmpose 1.3.2, run at
its native 640x640. The pose model is ``rtmpose-l body7`` COCO-17, a deliberate
step up from the alias's RTMPose-M body7 (see
docs/architecture_notes/rtmlib_migration/README.md).

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

from pipeline.config import COCO_N_JOINTS

# rtmlib-loadable mmdeploy ONNX-SDK archives. The detector is the ONNX export of
# the RTMDet-M person checkpoint (235e8209) that MMPoseInferencer("human")
# resolves at mmpose 1.3.2; the pose model is the updated body7 RTMPose-L.
# Pin + SHA-verify via validation_scripts/rtmlib_migration.
# The "256x192" in the pose filename is OpenMMLab's height x width naming (named
# HxW upstream); every resolution we write elsewhere is width x height (W x H).
_MODEL_BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
DET_URL = _MODEL_BASE + "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.zip"
POSE_URL = _MODEL_BASE + "rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.zip"
# rtmlib reads the detector size as (H, W) but the pose size as (W, H): RTMDet's
# preprocess pads a (size[0], size[1], 3) = (rows, cols) canvas, while RTMPose's
# top_down_affine unpacks w, h = input_size. 640x640 is square, so order is moot.
DET_INPUT_SIZE = (640, 640)   # rtmlib (H, W); native to the export
POSE_INPUT_SIZE = (192, 256)  # rtmlib (W, H); 192x256 (W x H), the 256x192 model
# Detector keep-filter, matching mmpose's cut: MMPoseInferencer applied a strict
# score > 0.3 to this detector's output (the committed raw's min bbox_score is
# 0.30008). See docs/architecture_notes/rtmlib_migration/README.md.
DET_SCORE_THR = 0.3


class FrameDetections(NamedTuple):
    """The ``n_people`` real (unpadded) detections in one frame, detector slot order.

    ``n_people`` is the count of people the detector actually found, one row per
    detected person (may be 0). NaN-padding and the N_max cap are a
    ``raw_extract`` concern; this carries only those real detections.

    ``J`` in the shape comments below is the COCO joint count (``COCO_N_JOINTS``,
    17), matching the RTMPose-L body7 model and the mmpose extract it replaces.
    """
    keypoints: np.ndarray    # (n_people, J, 2) float32; image-pixel coords, COCO-17 order
    bboxes: np.ndarray       # (n_people, 4) float32; xyxy image pixels
    bbox_scores: np.ndarray  # (n_people,) float32; per-person detection confidence
    kp_scores: np.ndarray    # (n_people, J) float32; per-joint confidence


class RTMDetScored(RTMDet):
    """``RTMDet`` that also returns the per-box detection score.

    The default person ONNX runs non-maximum suppression (NMS) inside the model
    graph, so its output is already the final boxes with scores, shape
    ``(1, N, 5)``: box in columns ``:4``, score in column 4. Stock
    ``RTMDet.postprocess`` returns only the boxes and drops that score; we need
    it downstream (``_raw_scores`` -> ``sticky_anchor``), so we read it back and
    filter on ``self.score_thr``.

    An export without NMS in the graph would instead emit raw per-anchor
    predictions needing grid-decode and suppression in Python, which this
    adapter does not implement: hence the loud failure below on any non-5-wide
    output.
    """

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image, ratio = self.preprocess(image)
        outputs = self.inference(image)[0]
        if outputs.shape[-1] != 5:
            raise RuntimeError(
                "expected a detector output with a score column (..., 5); got "
                f"{outputs.shape}. The default rtmdet-m-person ONNX runs NMS inside "
                "the graph and emits final boxes with scores; an export without it "
                "emits raw per-anchor predictions that would need grid-decoding and "
                "suppression in Python, which this adapter does not implement."
            )
        boxes = (outputs[0, :, :4] / ratio).astype(np.float32)  # (N, 4) xyxy, orig pixels
        scores = outputs[0, :, 4].astype(np.float32)            # (N,)
        # Stock's NMS branch filters at a hardcoded literal 0.3 and never reads
        # self.score_thr (only the non-NMS branch, which this detector never hits,
        # consults it). Applying it here (default 0.3) makes the cut configurable
        # while matching stock exactly at the default.
        keep = scores > self.score_thr
        return boxes[keep], scores[keep]


class RtmlibPoseExtractor:
    """Loads the detector + pose models once and extracts per-frame detections.

    CPU inference is deterministic run-to-run at a fixed thread count. rtmlib's
    ``BaseTool`` builds the onnxruntime session with no ``SessionOptions``, so
    ``intra_op_num_threads`` is not a constructor knob; pin threads via
    ``OMP_NUM_THREADS`` in the environment when a caller needs bit-reproducibility
    (the CPU determinism gate does). CUDA is nondeterministic regardless.

    :param device: onnxruntime device, ``"cpu"`` or ``"cuda"`` (needs
        ``onnxruntime-gpu`` for the latter).
    :param det_url: person-detector ONNX (defaults to the RTMDet-M person export
        of the checkpoint mmpose's inferencer used).
    :param pose_url: pose ONNX (defaults to RTMPose-L body7 COCO-17, 192x256 (W x H)).
    :param det_input_size: detector input in rtmlib's (H, W) order; 640x640 for the
        default ONNX (square, so order is moot).
    :param pose_input_size: pose input in rtmlib's (W, H) order; 192x256 (W x H) for
        the default 256x192 model.
    :param det_score_thr: keep detections scoring above this (default 0.3, mmpose's
        cut; see the module ``DET_SCORE_THR`` comment).
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
        """Run detector + pose on one BGR frame; return the ``n_people`` real detections.

        On zero detections we return empty ``(n_people=0)`` arrays rather than let
        RTMPose fall back to a whole-image "person" (its stock empty-bbox
        behaviour, which would fabricate a spurious detection).

        :param frame_bgr: (H, W, 3) uint8 BGR image (OpenCV convention).
        :return: ``FrameDetections`` for the ``n_people`` people found.
        """
        boxes, bbox_scores = self.det(frame_bgr)  # (n_people, 4), (n_people,)
        if len(boxes) == 0:
            return FrameDetections(
                keypoints=np.empty((0, COCO_N_JOINTS, 2), dtype=np.float32),
                bboxes=np.empty((0, 4), dtype=np.float32),
                bbox_scores=np.empty((0,), dtype=np.float32),
                kp_scores=np.empty((0, COCO_N_JOINTS), dtype=np.float32),
            )
        # RGB fix: RTMPose's mean is RGB-order but it never converts, so hand it RGB.
        # warpAffine is colour-agnostic, so convert-then-warp == mmpose's warp-then-convert.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        keypoints, kp_scores = self.pose(frame_rgb, bboxes=boxes)  # (n_people, J, 2), (n_people, J)
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
