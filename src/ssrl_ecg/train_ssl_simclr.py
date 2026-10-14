"""Shared repaired simclr training entry point."""

from ssrl_ecg.cli import run_training_cli


def main():
    run_training_cli("simclr")


if __name__ == "__main__":
    main()
