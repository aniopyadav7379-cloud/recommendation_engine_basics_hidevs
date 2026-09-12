import pytest

from reco_engine.cli import main
from reco_engine.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RECO_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestCli:
    def test_recommend_for_unknown_user_prints_no_results_message(self, capsys):
        exit_code = main(["recommend", "cold_user"])
        assert exit_code == 0
        assert "No recommendations available" in capsys.readouterr().out

    def test_record_then_recommend_round_trip(self, capsys, tmp_path):
        (tmp_path / "item_features.json").write_text(
            '{"job_1": ["python"], "job_2": ["python", "ml"]}'
        )
        record_exit = main(["record", "alice", "job_1"])
        assert record_exit == 0
        assert "Recorded: alice -> job_1" in capsys.readouterr().out

        recommend_exit = main(["recommend", "alice", "--k", "1"])
        assert recommend_exit == 0

    def test_missing_command_is_required(self):
        with pytest.raises(SystemExit):
            main([])
