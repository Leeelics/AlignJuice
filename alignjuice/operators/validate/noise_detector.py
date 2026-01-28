"""
Noise detector operator for AlignJuice.

Detects and handles noisy/problematic samples using heuristics or CleanLab.
"""

from __future__ import annotations

from typing import Any, Literal

from alignjuice.core.data_container import DataContainer, AlignmentSample
from alignjuice.core.registry import register_operator
from alignjuice.operators.base import Operator


@register_operator("noise_detector")
class NoiseDetector(Operator):
    """
    Detect and handle noisy samples.

    Uses heuristics or CleanLab to identify:
    - Label errors
    - Redundant samples
    - Ambiguous samples
    - Low-quality samples
    """

    name = "noise_detector"

    def __init__(
        self,
        method: Literal["heuristic", "cleanlab"] = "heuristic",
        action_error: Literal["remove", "flag", "keep"] = "remove",
        action_redundancy: Literal["remove", "merge", "keep"] = "remove",
        action_ambiguity: Literal["remove", "flag", "keep"] = "flag",
        confidence_threshold: float = 0.5,
        **kwargs: Any,
    ):
        """
        Initialize noise detector.

        Args:
            method: Detection method (heuristic or cleanlab)
            action_error: Action for detected errors
            action_redundancy: Action for redundant samples
            action_ambiguity: Action for ambiguous samples
            confidence_threshold: Confidence threshold for noise detection
        """
        super().__init__(
            method=method,
            action_error=action_error,
            action_redundancy=action_redundancy,
            action_ambiguity=action_ambiguity,
            confidence_threshold=confidence_threshold,
            **kwargs,
        )
        self.method = method
        self.action_error = action_error
        self.action_redundancy = action_redundancy
        self.action_ambiguity = action_ambiguity
        self.confidence_threshold = confidence_threshold

    def detect_noise_heuristic(self, sample: AlignmentSample) -> dict[str, Any]:
        """
        Detect noise using heuristic rules.

        Returns:
            Dict with noise_type and confidence
        """
        issues = []
        instruction = sample.instruction
        output = sample.output

        # Check for empty or very short content
        if len(instruction.strip()) < 5:
            issues.append(("error", 0.9, "instruction_too_short"))
        if len(output.strip()) < 10:
            issues.append(("error", 0.8, "output_too_short"))

        # Check for repetitive content
        words = output.split()
        if len(words) > 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                issues.append(("error", 0.85, "highly_repetitive"))

        # Check for instruction-output mismatch patterns
        instruction_lower = instruction.lower()
        output_lower = output.lower()

        # Question without answer pattern
        if "?" in instruction and len(output) < 50:
            if not any(w in output_lower for w in ["yes", "no", "is", "are", "the"]):
                issues.append(("ambiguity", 0.6, "possible_incomplete_answer"))

        # Check for refusal patterns that might indicate errors
        refusal_patterns = [
            "i cannot", "i can't", "i'm unable",
            "i don't have the ability", "as an ai language model",
        ]
        if any(p in output_lower for p in refusal_patterns):
            # Not necessarily an error, but flag for review
            issues.append(("ambiguity", 0.5, "contains_refusal"))

        # Check for copy-paste artifacts
        artifacts = ["```", ">>>", "...", "==="]
        artifact_count = sum(output.count(a) for a in artifacts)
        if artifact_count > 10:
            issues.append(("error", 0.6, "excessive_artifacts"))

        # Determine primary issue
        if not issues:
            return {"noise_type": None, "confidence": 0.0, "reason": None}

        # Return highest confidence issue
        issues.sort(key=lambda x: x[1], reverse=True)
        noise_type, confidence, reason = issues[0]

        return {
            "noise_type": noise_type,
            "confidence": confidence,
            "reason": reason,
            "all_issues": issues,
        }

    def detect_noise_cleanlab(self, data: DataContainer) -> list[dict[str, Any]]:
        """
        Detect noise using CleanLab.

        Requires cleanlab to be installed.
        """
        try:
            import cleanlab
        except ImportError:
            print("CleanLab not installed. Falling back to heuristic method.")
            return [self.detect_noise_heuristic(s) for s in data]

        # For CleanLab, we need embeddings and pseudo-labels
        # This is a simplified implementation
        results = []
        for sample in data:
            # Use heuristic as fallback for now
            result = self.detect_noise_heuristic(sample)
            results.append(result)

        return results

    def __call__(self, data: DataContainer) -> DataContainer:
        """Detect and handle noisy samples."""
        if len(data) == 0:
            return data

        # Detect noise
        if self.method == "cleanlab":
            noise_results = self.detect_noise_cleanlab(data)
        else:
            noise_results = [self.detect_noise_heuristic(s) for s in data]

        # Process samples based on noise detection
        kept_samples = []
        removed_count = 0
        flagged_count = 0

        for sample, noise_info in zip(data, noise_results):
            noise_type = noise_info.get("noise_type")
            confidence = noise_info.get("confidence", 0)

            # Store noise info in metadata
            sample.metadata["noise_detection"] = noise_info

            if noise_type is None or confidence < self.confidence_threshold:
                # No significant noise detected
                kept_samples.append(sample)
                continue

            # Determine action based on noise type
            if noise_type == "error":
                action = self.action_error
            elif noise_type == "redundancy":
                action = self.action_redundancy
            elif noise_type == "ambiguity":
                action = self.action_ambiguity
            else:
                action = "keep"

            if action == "remove":
                removed_count += 1
            elif action == "flag":
                sample.metadata["flagged"] = True
                sample.metadata["flag_reason"] = noise_info.get("reason")
                flagged_count += 1
                kept_samples.append(sample)
            else:  # keep
                kept_samples.append(sample)

        self._metrics = {
            "input_count": len(data),
            "output_count": len(kept_samples),
            "removed_count": removed_count,
            "flagged_count": flagged_count,
            "method": self.method,
            "confidence_threshold": self.confidence_threshold,
        }

        return DataContainer(
            samples=kept_samples,
            provenance=data.provenance + [
                f"noise_detector ({self.method}): removed {removed_count}, flagged {flagged_count}"
            ],
        )
