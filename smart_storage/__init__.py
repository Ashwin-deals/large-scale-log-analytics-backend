"""Smart storage: decide which logs to keep.

Every block gets an importance score; the score places it in a HOT / WARM /
COLD tier, and the tier decides how many of its log lines are retained. A
genetic algorithm tunes the scoring weights and tier thresholds to save as
much storage as possible while still keeping the anomalies.

  policy.py     -- scoring, tiering, storage accounting (pure numpy, no I/O)
  baselines.py  -- the policies the tuned one has to beat
  optimizer.py  -- PyGAD search, stratified tune/holdout split, report builder
  data.py       -- scores the labeled blocks with the deployed model
"""
