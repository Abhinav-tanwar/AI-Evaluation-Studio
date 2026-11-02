import argparse
import json
from pathlib import Path

from .evaluation import run_comparison
from .schemas import RunRequest
from .settings import Settings
from .storage import Store


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate two RAG configurations and apply a regression gate."
    )
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--provider", choices=["reference", "openai"], default="reference")
    parser.add_argument("--output", type=Path, default=Path("reports/latest.json"))
    parser.add_argument("--gate", action="store_true", help="Exit with status 1 if a regression gate fails.")
    parser.add_argument("--save", action="store_true", help="Save this result to the application database.")
    args = parser.parse_args()
    settings = Settings()
    result = run_comparison(
        RunRequest(name=f"{args.split.title()} benchmark", split=args.split, provider=args.provider), settings
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.save:
        store = Store(settings.database_url)
        store.put({**result, "status": "completed"})
        store.engine.dispose()
    for label, pipeline in result["pipelines"].items():
        print(label, json.dumps(pipeline["metrics"]))
    print("Regression gate:", "PASS" if result["gate"]["passed"] else "FAIL")
    if args.gate and not result["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
