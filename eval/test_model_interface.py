"""Validation checks for the shared PathoReason prediction interface."""

from models.base import PredictionResult


def main() -> None:
    track_a_result = PredictionResult(
        label="SSA",
        confidence=0.91,
        explanation=None,
        evidence=(),
        model_name="PLIP",
    )

    track_b_result = PredictionResult(
        label="HP",
        confidence=None,
        explanation="The image shows relatively regular crypt architecture.",
        evidence=("regular crypt architecture",),
        model_name="GPT-4o",
    )

    assert track_a_result.label == "SSA"
    assert track_a_result.confidence == 0.91

    assert track_b_result.label == "HP"
    assert track_b_result.confidence is None
    assert len(track_b_result.evidence) == 1

    print("Shared model interface validation passed.")


if __name__ == "__main__":
    main()
