"""Generate synthetic notes: python scripts/generate_data.py data/raw/... data/processed."""
import sys

from carebot.data_gen import generate


if __name__ == "__main__":
    print(generate(sys.argv[1], sys.argv[2]))
