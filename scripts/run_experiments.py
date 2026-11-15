"""Run Week 6 experiments and save a JSON result."""
import json
import sys
from pathlib import Path

from carebot.experiments import run


if __name__ == "__main__":
    result = run(sys.argv[1], sys.argv[2], sys.argv[3])
    Path("docs/experiments_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
