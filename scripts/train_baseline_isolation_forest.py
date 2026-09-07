import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from detection.evaluate import evaluate, print_metrics_table
from detection.predict import predict
from detection.train import train


def main() -> None:
    model, merged = train()
    predictions = predict(model, merged)
    metrics = evaluate(predictions)
    print_metrics_table(metrics)


if __name__ == "__main__":
    main()
