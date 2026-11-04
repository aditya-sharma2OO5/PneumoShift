"""Unit tests for Phase 6 Results Aggregation configuration schemas and data models."""

from pathlib import Path
import pytest
import yaml

from pneumoshift.aggregation.config_schemas import (
    AggregationConfig,
    BootstrapCIResult,
    HypothesisTestRecord,
    MasterTableSpec,
    load_aggregation_config,
    validate_aggregation_config,
    validate_bootstrap_config,
    validate_export_config,
    validate_hypothesis_testing_config,
    validate_hypothesis_tests_spec,
    validate_master_tables_spec,
    VALID_ALTERNATIVES,
    VALID_BOOTSTRAP_METHODS,
    VALID_CORRECTION_METHODS,
    VALID_EXPORT_FORMATS,
    VALID_TEST_TYPES,
)

CONFIGS_DIR = Path("configs/aggregation")


class TestPhase6ConfigLoading:
    """Verifies that all Phase 6 YAML files in configs/aggregation/ pass schema validation."""

    def test_default_aggregation_yaml_valid(self):
        cfg_path = CONFIGS_DIR / "default.yaml"
        assert cfg_path.is_file(), f"Missing config file: {cfg_path}"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        validated = validate_aggregation_config(cfg)
        assert validated["experiment_name"] == "master_results_aggregation"
        assert validated["bootstrap"]["method"] in VALID_BOOTSTRAP_METHODS
        assert validated["hypothesis_testing"]["fdr_alpha"] == 0.05
        assert all(f in VALID_EXPORT_FORMATS for f in validated["exports"]["formats"])

    def test_hypothesis_tests_yaml_valid(self):
        cfg_path = CONFIGS_DIR / "hypothesis_tests.yaml"
        assert cfg_path.is_file(), f"Missing config file: {cfg_path}"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        validated = validate_hypothesis_tests_spec(cfg)
        assert len(validated["hypotheses"]) >= 5
        ids = [h["id"] for h in validated["hypotheses"]]
        assert "H1" in ids
        assert "H2" in ids
        assert "H3" in ids
        assert "H4" in ids
        assert "H5" in ids

    def test_master_tables_yaml_valid(self):
        cfg_path = CONFIGS_DIR / "master_tables.yaml"
        assert cfg_path.is_file(), f"Missing config file: {cfg_path}"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        validated = validate_master_tables_spec(cfg)
        assert len(validated["tables"]) == 6
        table_ids = [t["id"] for t in validated["tables"]]
        assert "table1" in table_ids
        assert "table6" in table_ids

    def test_load_aggregation_config_helper(self):
        cfg = load_aggregation_config(CONFIGS_DIR / "default.yaml")
        assert cfg["experiment_name"] == "master_results_aggregation"

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_aggregation_config(CONFIGS_DIR / "nonexistent.yaml")


class TestBootstrapConfigValidation:
    """Tests boundary assertions for bootstrap parameters."""

    def test_valid_minimal_bootstrap_config(self):
        cfg = {"n_resamples": 500, "confidence_level": 0.90, "seed": 123, "method": "patient_clustered"}
        val = validate_bootstrap_config(cfg)
        assert val["n_resamples"] == 500

    def test_low_resamples_raises(self):
        with pytest.raises(ValueError, match="n_resamples must be an integer >= 10"):
            validate_bootstrap_config({"n_resamples": 5})

    def test_invalid_confidence_level_raises(self):
        with pytest.raises(ValueError, match="confidence_level must be in"):
            validate_bootstrap_config({"confidence_level": 1.5})
        with pytest.raises(ValueError, match="confidence_level must be in"):
            validate_bootstrap_config({"confidence_level": -0.1})

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Invalid bootstrap method"):
            validate_bootstrap_config({"method": "unknown_method"})

    def test_empty_cluster_by_raises(self):
        with pytest.raises(ValueError, match="cluster_by must be a non-empty string"):
            validate_bootstrap_config({"cluster_by": "   "})

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError, match="seed must be a non-negative integer"):
            validate_bootstrap_config({"seed": -1})


class TestHypothesisTestingConfigValidation:
    """Tests boundary assertions for hypothesis testing parameters."""

    def test_valid_hypothesis_testing_config(self):
        cfg = {
            "fdr_alpha": 0.05,
            "family_wise_alpha": 0.05,
            "correction_methods": ["benjamini_hochberg", "bonferroni"],
            "default_test_type": "paired_bootstrap",
        }
        val = validate_hypothesis_testing_config(cfg)
        assert val["fdr_alpha"] == 0.05

    def test_invalid_fdr_alpha_raises(self):
        with pytest.raises(ValueError, match="fdr_alpha must be in"):
            validate_hypothesis_testing_config({"fdr_alpha": 1.0})
        with pytest.raises(ValueError, match="fdr_alpha must be in"):
            validate_hypothesis_testing_config({"fdr_alpha": 0.0})

    def test_invalid_correction_method_raises(self):
        with pytest.raises(ValueError, match="Invalid correction method"):
            validate_hypothesis_testing_config({"correction_methods": ["unknown_correction"]})

    def test_invalid_default_test_type_raises(self):
        with pytest.raises(ValueError, match="Invalid default_test_type"):
            validate_hypothesis_testing_config({"default_test_type": "unknown_test"})


class TestExportConfigValidation:
    """Tests boundary assertions for table exports."""

    def test_valid_export_config(self):
        cfg = {"formats": ["csv", "md"], "precision": 3, "latex_booktabs": True}
        val = validate_export_config(cfg)
        assert val["precision"] == 3

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid export format"):
            validate_export_config({"formats": ["docx"]})

    def test_negative_precision_raises(self):
        with pytest.raises(ValueError, match="precision must be a non-negative integer"):
            validate_export_config({"precision": -1})

    def test_non_bool_booktabs_raises(self):
        with pytest.raises(ValueError, match="latex_booktabs must be a boolean"):
            validate_export_config({"latex_booktabs": "yes"})


class TestDataclassesAndSpecs:
    """Tests dataclass initialization, bounds, and methods."""

    def test_bootstrap_ci_result_valid(self):
        res = BootstrapCIResult(
            point_estimate=0.768,
            ci_lower=0.752,
            ci_upper=0.784,
            standard_error=0.008,
            n_resamples=1000,
            confidence_level=0.95,
        )
        assert res.format_ci(3) == "0.768 [0.752, 0.784]"
        d = res.to_dict()
        assert d["point_estimate"] == 0.768

    def test_bootstrap_ci_result_invalid_bounds_raises(self):
        with pytest.raises(ValueError, match="ci_lower .* cannot be strictly greater than ci_upper"):
            BootstrapCIResult(point_estimate=0.7, ci_lower=0.8, ci_upper=0.6)

    def test_hypothesis_test_record_valid(self):
        rec = HypothesisTestRecord(
            hypothesis_id="H1",
            name="Generalization_Drop",
            null_hypothesis="Delta_AUROC <= 0",
            alternative="greater",
            test_type="paired_bootstrap",
            effect_size=-0.107,
            raw_p_value=0.0001,
            fdr_q_value=0.0005,
            is_significant=True,
        )
        assert rec.is_significant is True
        d = rec.to_dict()
        assert d["hypothesis_id"] == "H1"

    def test_hypothesis_test_record_invalid_p_value_raises(self):
        with pytest.raises(ValueError, match="raw_p_value must be in"):
            HypothesisTestRecord(
                hypothesis_id="H1",
                name="x",
                null_hypothesis="x",
                alternative="greater",
                test_type="paired_bootstrap",
                effect_size=0.1,
                raw_p_value=1.5,
                fdr_q_value=0.1,
                is_significant=False,
            )

    def test_aggregation_config_dataclass(self):
        cfg = AggregationConfig(experiment_name="custom_aggregation")
        assert cfg.experiment_name == "custom_aggregation"
        d = cfg.to_dict()
        assert d["experiment_name"] == "custom_aggregation"

        with pytest.raises(ValueError, match="experiment_name must be a non-empty string"):
            AggregationConfig(experiment_name="   ")

    def test_master_table_spec_dataclass(self):
        spec = MasterTableSpec(
            id="table1",
            name="test_table",
            title="Table 1 Test",
            description="Test table description",
            columns=["Col A", "Col B"],
        )
        assert spec.id == "table1"
        assert len(spec.columns) == 2
        d = spec.to_dict()
        assert d["title"] == "Table 1 Test"

    def test_duplicate_hypothesis_id_raises(self):
        cfg = {
            "hypotheses": [
                {"id": "H1", "name": "A", "null_hypothesis": "A <= 0", "test_type": "paired_bootstrap"},
                {"id": "H1", "name": "B", "null_hypothesis": "B <= 0", "test_type": "paired_bootstrap"},
            ]
        }
        with pytest.raises(ValueError, match="Duplicate hypothesis id detected: 'H1'"):
            validate_hypothesis_tests_spec(cfg)

    def test_duplicate_table_id_raises(self):
        cfg = {
            "tables": [
                {"id": "T1", "title": "Table 1", "columns": ["A", "B"]},
                {"id": "T1", "title": "Table 1 Dup", "columns": ["C", "D"]},
            ]
        }
        with pytest.raises(ValueError, match="Duplicate table id detected: 'T1'"):
            validate_master_tables_spec(cfg)
