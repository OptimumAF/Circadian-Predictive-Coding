"""Run the fixed local coupled pruning/consolidation/promotion/rollback example."""

import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.adapters.coupled_topology_demo import describe_example, encode_record, run_example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="Describe without model work")
    args = parser.parse_args()
    encoded = encode_record(describe_example() if args.describe else run_example(), indent=2)
    if len(encoded.encode("utf-8")) > 2 * 1024**2:
        raise RuntimeError("engineering example exceeds 2MiB output allowance")
    print(encoded)


if __name__ == "__main__":
    main()
