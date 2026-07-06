"""Commentary-scraper package: stages of the scrape-to-dataset pipeline.

Built from local_scratch/autograder_architecture/scraper_spec.md (B2). The
config module is the single source of truth for file contracts and named
constants; each stage module is runnable as `python -m scraper.<module>` with
PYTHONPATH=src. Stage numbering follows pipeline_stage_map.md.
"""
