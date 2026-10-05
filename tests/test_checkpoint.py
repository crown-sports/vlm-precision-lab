from copy import deepcopy

import pytest

from precisionlab.checkpoint import legacy_null_dtype_fields


def test_legacy_adapter_removes_only_null_metadata_and_preserves_original():
    config = {"quantization_config": {"config_groups": {"g": {
        "weights": {"num_bits": 4, "group_size": 128, "scale_dtype": None, "zp_dtype": None},
        "input_activations": None, "output_activations": None}}}, "model_type": "qwen3_vl"}
    original = deepcopy(config)
    converted, removed = legacy_null_dtype_fields(config)
    assert config == original
    assert converted["quantization_config"]["config_groups"]["g"]["weights"] == {"num_bits": 4, "group_size": 128}
    assert len(removed) == 2 and converted["model_type"] == "qwen3_vl"


def test_legacy_adapter_refuses_to_guess_a_real_scale_precision():
    config = {"quantization_config": {"config_groups": {"g": {"weights": {"scale_dtype": "float16"}}}}}
    with pytest.raises(ValueError, match="Non-null dtype"):
        legacy_null_dtype_fields(config)
