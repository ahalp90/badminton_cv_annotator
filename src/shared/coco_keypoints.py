"""The COCO-17 keypoint layout that RTMPose predicts.

This module holds plain constants only, so pose consumers can share them
without importing RTMLib or BST-X.
"""

# Joints per person. Every pose array's joint axis has this size.
COCO_N_JOINTS = 17
