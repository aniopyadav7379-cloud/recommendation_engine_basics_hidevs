import pytest

from reco_engine.config import Settings, get_settings
from reco_engine.exceptions import ConfigurationError


class TestSettingsValidation:
    def test_defaults_are_valid(self):
        settings = Settings()
        assert settings.max_candidates == 20
        assert settings.default_top_k == 10
        assert settings.max_k == 100
        assert settings.log_level == "INFO"

    def test_negative_max_candidates_raises(self):
        with pytest.raises(ConfigurationError):
            Settings(max_candidates=0)

    def test_negative_default_top_k_raises(self):
        with pytest.raises(ConfigurationError):
            Settings(default_top_k=-1)

    def test_zero_max_k_raises(self):
        with pytest.raises(ConfigurationError):
            Settings(max_k=0)

    def test_default_top_k_above_max_k_raises(self):
        with pytest.raises(ConfigurationError):
            Settings(default_top_k=50, max_k=10)

    def test_negative_weight_raises(self):
        with pytest.raises(ConfigurationError):
            Settings(weight_relevance=-0.1)

    def test_all_zero_weights_raises(self):
        with pytest.raises(ConfigurationError):
            Settings(weight_relevance=0, weight_recency=0, weight_popularity=0)

    def test_single_positive_weight_is_sufficient(self):
        settings = Settings(weight_relevance=1.0, weight_recency=0, weight_popularity=0)
        assert settings.weight_relevance == 1.0

    def test_port_zero_raises(self):
        with pytest.raises(ConfigurationError):
            Settings(api_port=0)

    def test_port_above_65535_raises(self):
        with pytest.raises(ConfigurationError):
            Settings(api_port=70000)

    def test_port_boundaries_are_valid(self):
        assert Settings(api_port=1).api_port == 1
        assert Settings(api_port=65535).api_port == 65535

    def test_log_level_is_normalized_to_uppercase(self):
        assert Settings(log_level="debug").log_level == "DEBUG"
        assert Settings(log_level=" Warning ").log_level == "WARNING"

    def test_invalid_log_level_raises(self):
        with pytest.raises(ConfigurationError):
            Settings(log_level="not-a-level")

    def test_env_vars_are_preserved(self, monkeypatch: pytest.MonkeyPatch):
        """Every pre-existing RECO_* variable must still be honored."""
        monkeypatch.setenv("RECO_MAX_CANDIDATES", "42")
        monkeypatch.setenv("RECO_WEIGHT_RELEVANCE", "0.1")
        monkeypatch.setenv("RECO_WEIGHT_RECENCY", "0.2")
        monkeypatch.setenv("RECO_WEIGHT_POPULARITY", "0.3")
        monkeypatch.setenv("RECO_DEFAULT_TOP_K", "7")
        monkeypatch.setenv("RECO_DATA_DIR", "custom_data")
        monkeypatch.setenv("RECO_HISTORY_FILE", "hist.json")
        monkeypatch.setenv("RECO_FEATURES_FILE", "feat.json")
        monkeypatch.setenv("RECO_LOG_LEVEL", "debug")
        monkeypatch.setenv("RECO_API_HOST", "127.0.0.1")
        monkeypatch.setenv("RECO_API_PORT", "9000")
        get_settings.cache_clear()
        try:
            settings = get_settings()
            assert settings.max_candidates == 42
            assert settings.weight_relevance == 0.1
            assert settings.default_top_k == 7
            assert settings.data_dir == "custom_data"
            assert settings.history_file == "hist.json"
            assert settings.features_file == "feat.json"
            assert settings.log_level == "DEBUG"
            assert settings.api_host == "127.0.0.1"
            assert settings.api_port == 9000
        finally:
            get_settings.cache_clear()

    def test_new_max_k_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RECO_MAX_K", "25")
        get_settings.cache_clear()
        try:
            assert get_settings().max_k == 25
        finally:
            get_settings.cache_clear()
