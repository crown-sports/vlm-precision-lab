"""Narrow metadata adapter for compressed-tensors 0.13 exports / 0.11 readers."""

from copy import deepcopy


def legacy_null_dtype_fields(config):
    """Remove only unsupported null dtype declarations, never quantized values."""
    result = deepcopy(config)
    removed = []
    groups = result.get("quantization_config", {}).get("config_groups", {})
    for name, scheme in groups.items():
        for component in ("weights", "input_activations", "output_activations"):
            args = scheme.get(component)
            if not isinstance(args, dict):
                continue
            for key in ("scale_dtype", "zp_dtype"):
                if key in args:
                    if args[key] is not None:
                        raise ValueError("Non-null dtype metadata requires a compatible reader")
                    del args[key]
                    removed.append(f"quantization_config.config_groups.{name}.{component}.{key}")
    return result, removed
