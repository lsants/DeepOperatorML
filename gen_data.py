import os
import sys
import argparse
import logging
from src.problems import ProblemRegistry
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

def gen_data() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, type=str, help="Problem name")
    args = parser.parse_args()

    config_path = os.path.join("./configs/problems/", args.problem, 'datagen.yaml')

    try:
        generator = ProblemRegistry.get_generator(name=args.problem, config=config_path)
        generator.generate()
    except KeyError:
        logging.error(f"Unknown problem: {args.problem}")
        raise

if __name__ == "__main__":
    gen_data()