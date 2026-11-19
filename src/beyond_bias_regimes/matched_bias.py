"""Gains from leakage at matched bias (Section 8.4): the cheapest traps of Section 8.3 at one
accuracy eps for all alpha at each b*. The iteration costs are those of Section 8.2."""

import argparse

from .noise_tolerance import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/matched_bias.toml")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
