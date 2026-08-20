from experiments.analyse_broadcast_priors import rule_metrics


def test_rule_metrics_reports_precision_recall_and_rejection() -> None:
    rows = [
        {"priors": {"signal": 1.0}, "truth": {"valid_rally": True}},
        {"priors": {"signal": 0.8}, "truth": {"valid_rally": True}},
        {"priors": {"signal": 0.7}, "truth": {"valid_rally": False}},
        {"priors": {"signal": 0.2}, "truth": {"valid_rally": False}},
    ]

    metrics = rule_metrics(rows, "signal", 0.75, keep_when_high=True)

    assert metrics["keep_precision"] == 1.0
    assert metrics["live_recall"] == 1.0
    assert metrics["nonlive_rejection"] == 1.0
