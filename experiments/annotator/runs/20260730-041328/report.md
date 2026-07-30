# Annotator run current_annotator_8config_288p

Outcome: succeeded; elapsed: 1450.3 seconds.
Device: requested cuda, resolved cuda.  Source commit: 189c5af58e45d23ae827dde516924194eb238e18.

| configuration | rally coverage | contact P/R/F1 | +/-5 P/R/F1 | +/-10 P/R/F1 | court valid |
| --- | --- | --- | --- | --- | --- |
| static_shuttleset_homography/sset_01/tracknet-stride-8 | 0.973 | 0.640/0.679/0.659 | 0.624/0.679/0.650 | 0.719/0.782/0.749 | 0.331 |
| static_shuttleset_homography/sset_01/tracknet-stride-1 | 0.814 | 0.637/0.498/0.559 | 0.767/0.498/0.604 | 0.834/0.542/0.657 | 0.331 |
| static_shuttleset_homography/sset_15/tracknet-stride-8 | 0.808 | 0.495/0.712/0.584 | 0.675/0.712/0.693 | 0.736/0.777/0.756 | 0.264 |
| static_shuttleset_homography/sset_21/tracknet-stride-8 | 0.733 | 0.417/0.640/0.504 | 0.521/0.640/0.574 | 0.563/0.691/0.620 | 0.372 |
| detected_ckn_opencv_consensus/sset_01/tracknet-stride-8 | 0.973 | 0.654/0.679/0.666 | 0.615/0.679/0.646 | 0.709/0.782/0.744 | 0.324 |
| detected_ckn_opencv_consensus/sset_01/tracknet-stride-1 | 0.814 | 0.646/0.498/0.563 | 0.771/0.498/0.605 | 0.838/0.542/0.658 | 0.324 |
| detected_ckn_opencv_consensus/sset_15/tracknet-stride-8 | 0.798 | 0.573/0.712/0.635 | 0.707/0.712/0.710 | 0.770/0.775/0.773 | 0.214 |
| detected_ckn_opencv_consensus/sset_21/tracknet-stride-8 | 0.640 | 0.443/0.584/0.504 | 0.575/0.584/0.579 | 0.621/0.630/0.626 | 0.320 |

Live CourtKeyNet/OpenCV detection is the operational default. Static homography is the controlled reference and manual fixed-camera fallback.
Ignored masks and arrays: 32 NPY files (4.3 MiB). Git will not preserve them; copy or archive them if wanted.
