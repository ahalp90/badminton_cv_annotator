from experiments.analyse_event_consensus import decision_metrics


def test_decision_metrics_counts_false_keeps_and_unresolved_recall() -> None:
    metrics = decision_metrics(
        decisions=[True, True, False, False],
        truth=[True, False, True, False],
    )

    assert metrics["true_kept"] == 1
    assert metrics["false_kept"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["precision_95ci"] is not None
