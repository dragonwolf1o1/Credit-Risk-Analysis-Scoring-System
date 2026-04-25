"""Train the credit risk model."""

from credit_risk.training import print_training_summary, train_and_save_model


def main() -> int:
    report = train_and_save_model()
    print_training_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
