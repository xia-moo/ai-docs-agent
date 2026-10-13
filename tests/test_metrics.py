import pytest
from ai_docs_agent.metrics import rank_metrics


def test_hand_calculated_metrics():
    metrics = rank_metrics(["wrong", "right"], {"right"}, 5)
    assert metrics["hit"] == 1
    assert metrics["recall"] == 1
    assert metrics["mrr"] == 0.5
    assert metrics["ndcg"] == pytest.approx(0.63092975357)
    assert rank_metrics(["wrong"], {"right"})["hit"] == 0
