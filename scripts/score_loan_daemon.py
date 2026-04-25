"""Continuously score pending loans."""

from credit_risk.scoring import run_daemon


def main() -> int:
    run_daemon()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
