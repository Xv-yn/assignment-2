"""Compatibility imports and CLI for the original CPU pilot entry point.

Configuration loading, argument handling and data auditing live in
run_experiments. The compatibility command retains its CPU-only guard and
does not start formal training; use algorithm.run_experiments for that stage.
"""

from .run_experiments import (
    main as _main,
    read_config,
    resolve_path,
    reusable_record,
    validate_pilot_config,
)


def main(argv=None):
    return _main(argv, pilot_only=True)


if __name__ == "__main__":
    main()
