"""The documented public API is importable from the package root."""

import vsbench


def test_public_api_exports():
    for name in vsbench.__all__:
        assert hasattr(vsbench, name), name
    assert callable(vsbench.run) and callable(vsbench.write_outputs)


def test_best_picks_fastest_config_meeting_recall(run_result):
    best = run_result.best(0.95)
    assert best is not None and best.ef_search == 40
    assert run_result.best(0.5).ef_search == 10  # faster of the two that qualify
    assert run_result.best(0.999) is None
