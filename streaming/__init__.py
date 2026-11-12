"""Real-time layer: Kafka in front of the existing batch models.

  config.py       topic names and connection defaults
  block_state.py  line -> event (same rules as parser + cleaner) and per-block
                  running state whose features match the batch builders exactly
  scoring.py      the early-warning, deployed and Smart Storage models
  processor.py    Kafka consumer: state, alerts, storage decisions, MongoDB sink
  producer.py     replays HDFS.log into logs.raw at a fixed rate

Nothing here changes the batch pipeline; it only reads its models and policy.
"""
