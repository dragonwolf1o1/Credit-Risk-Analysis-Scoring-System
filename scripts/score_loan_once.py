"""Score pending loans once."""

from credit_risk.scoring import print_scoring_summary, score_pending_loans


def main() -> int:
    result = score_pending_loans(source_script="score_loan_once")
    print_scoring_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
