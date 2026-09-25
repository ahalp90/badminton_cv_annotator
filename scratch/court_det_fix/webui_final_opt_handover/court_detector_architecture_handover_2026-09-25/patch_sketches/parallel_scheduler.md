# Deterministic scheduler sketch

## Pair tasks

```text
coordinator
  prepare immutable view features
  build ordered pair task list
  submit tasks to persistent process pool
  collect by pair_id
  restore task-list order
  concatenate per-pair retained arrays
  run current stable global retention
```

A worker returns arrays and counters only. It must not append to shared lists, mutate provenance maps, or write records.

## Parent/refit tasks

```text
coordinator
  canonicalise populations
  deduplicate homography bytes
  submit one measurement per unique key
  expand duplicate occurrences in original order
  build line maps once
  submit one refit per eligible parent
  restore parent order
  run current ranking/net selection serially
```

## Reproducibility checks

- force NumPy/OpenCV/BLAS threads to one inside every worker;
- use the same ordered pair and parent IDs as the serial path;
- stable-sort only in the coordinator;
- compare outputs at 1, 2, 4, 8, and maximum workers;
- randomise worker sleep/completion order in a test to prove order independence;
- keep a serial exact path in the same codebase.
