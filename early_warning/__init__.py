"""Early warning: flag a risky HDFS block from its first k events.

  prefix_features.py  one chunked pass over the event-level log -> per-block
                      features using only each block's first k events
  model.py            Isolation Forest per k, GA-tuned alert threshold,
                      held-out evaluation with lead time, trade-off selection
"""
