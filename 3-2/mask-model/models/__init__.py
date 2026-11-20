"""Current paired mask detector and compatible checkpoint loading."""
from .difference_gate_network import DifferenceGateMaskNet

__all__ = ["DifferenceGateMaskNet", "paired_model_from_checkpoint"]


def paired_model_from_checkpoint(checkpoint: dict):
    arguments = checkpoint.get("args", {})
    state = checkpoint.get("model", checkpoint)
    architecture = arguments.get("architecture")
    if architecture is None:
        if any(key.startswith("large_region_head.") for key in state):
            architecture = "difference_gate_scale"
        elif any(key.startswith("gates.") for key in state):
            architecture = "difference_gate"
    if architecture not in {"difference_gate", "difference_gate_scale"}:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture!r}; use the current paired detector.")
    model = DifferenceGateMaskNet(
        pretrained=False, large_region=architecture == "difference_gate_scale",
        source_decoder=arguments.get("source_decoder", "legacy"),
        source_refinement=arguments.get("source_refinement", False),
        pixel_refinement=arguments.get("pixel_refinement", False),
    )
    model.load_state_dict(state, strict=True)
    return model, architecture
