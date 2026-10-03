# Badminton court detector

The court detector locates the playing court in an image or video. It returns
four outer court corners in image pixels, or a reason that no court was found.
Those corners define a **homography**: a perspective mapping between the image
and the flat court floor. The annotation pipeline uses that mapping for court
positions and distances.

It combines pretrained line and person detection with the known badminton court
layout. It needs no separately trained court model or labelled examples from
each venue. The goal is court geometry that remains useful across unfamiliar
footage, including amateur recordings.

| Documentation | Purpose |
| --- | --- |
| [Usage guide](usage.md) | Setup, required inputs, examples, every CLI option and output formats |
| [Design](design.md) | Reasons for the current choices and constraints that maintenance must preserve |
| [Sampling and reuse](sampling_and_reuse.md) | Which video frames contribute to a court and how evidence moves between scenes |

## How it works

1. **Find lines.** DeepLSD detects straight line fragments. The detector searches
   for arrangements matching badminton markings, including the inner lines.
2. **Check the court in use.** Video detection normally checks where people
   stand over a short window. Camera geometry and player positions help reject
   plausible arrangements belonging to the wrong court or other objects.
3. **Choose and refine a fit.** Visible paint provides most of the score, with
   line geometry and net-post evidence contributing. The chosen markings are
   refitted to their painted stripes.
4. **Combine observations.** Video modes can combine markings from several
   frames and share courts between returning camera views. This helps when a
   marking is obscured in one frame and visible in another.

The output describes the outer doubles court, even during singles play. Each
video scene receives at most one fixed court projection. A scene is an interval
between camera cuts; a pan or zoom within that interval can make the projection
unsuitable for part of it.

## Inputs and practical limits

A fresh image run needs the image, a DeepLSD checkout and its weights. A video
run can use live detections or saved line and pose inputs. Player checks are on
by default for video, and optional for still images. The [usage guide](usage.md)
specifies the dependencies and file layouts for each route.

Accepted courts still need scrutiny on difficult footage. Advertising boards,
net tape and the wrong inner markings have produced false courts. Short scenes,
partial views and missed people can leave a court undetected. The internal
score measures support in the image; labelled comparisons and visual review
establish whether a result is useful.

## Development and existing results

The [development story](../../experiments/court_detector/report.md) explains why
the detector was built and what the experiments taught. The
[experiment overview](../../experiments/court_detector/README.md) links to the
comparisons and evidence. Those records describe the tested versions and footage.

The [released court dataset](../../data/court_detections/sset_and_sset22/extractions_20261003/README.md)
contains predictions for 86 ShuttleSet and ShuttleSet22 videos, with a loading
example. The [source README](../../src/court_detector/README.md#code-map) contains
the Python API, module map and relevant tests.
