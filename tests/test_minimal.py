"""
Minimal unit test for AlignJuice framework.

Run with: python -m pytest tests/test_minimal.py -v
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_data_container_creation():
    """Test DataContainer can be created and manipulated."""
    from alignjuice.core.data_container import DataContainer, AlignmentSample

    # Create samples
    samples = [
        AlignmentSample(
            id="test_001",
            instruction="What is 2+2?",
            input="",
            output="2+2 equals 4.",
            category="reasoning",
        ),
        AlignmentSample(
            id="test_002",
            instruction="Explain gravity.",
            input="",
            output="Gravity is a force that attracts objects with mass toward each other.",
            category="factual",
        ),
    ]

    # Create container
    data = DataContainer(samples=samples)

    assert len(data) == 2
    assert data[0].id == "test_001"
    assert data[1].category == "factual"
    print("✓ DataContainer creation test passed")


def test_data_container_from_list():
    """Test DataContainer.from_list method."""
    from alignjuice.core.data_container import DataContainer

    raw_data = [
        {
            "id": "s1",
            "instruction": "Hello",
            "input": "",
            "output": "Hi there!",
            "category": "daily",
        },
        {
            "id": "s2",
            "instruction": "Bye",
            "input": "",
            "output": "Goodbye!",
            "category": "daily",
        },
    ]

    data = DataContainer.from_list(raw_data)

    assert len(data) == 2
    assert data[0].instruction == "Hello"
    print("✓ DataContainer.from_list test passed")


def test_exact_dedup_operator():
    """Test ExactDedup operator."""
    from alignjuice.core.data_container import DataContainer
    from alignjuice.operators.dedup.exact import ExactDedup

    raw_data = [
        {"id": "1", "instruction": "What is AI?", "input": "", "output": "AI is...", "category": "factual"},
        {"id": "2", "instruction": "What is AI?", "input": "", "output": "Artificial Intelligence...", "category": "factual"},  # Duplicate instruction
        {"id": "3", "instruction": "What is ML?", "input": "", "output": "ML is...", "category": "factual"},
    ]

    data = DataContainer.from_list(raw_data)
    dedup = ExactDedup(field="instruction")
    result = dedup(data)

    assert len(result) == 2  # One duplicate removed
    assert dedup.metrics["removed_count"] == 1
    print("✓ ExactDedup operator test passed")


def test_quality_filter_operator():
    """Test QualityFilter operator."""
    from alignjuice.core.data_container import DataContainer
    from alignjuice.operators.filter.quality import QualityFilter

    raw_data = [
        {"id": "1", "instruction": "Explain quantum physics in detail.", "input": "", "output": "Quantum physics is a branch of physics that deals with phenomena at nanoscopic scales. It describes nature at the smallest scales of energy levels of atoms and subatomic particles.", "category": "factual"},
        {"id": "2", "instruction": "Hi", "input": "", "output": "Hello", "category": "daily"},  # Low quality (too short)
    ]

    data = DataContainer.from_list(raw_data)
    quality_filter = QualityFilter(threshold=0.6)
    result = quality_filter(data)

    # At least one should pass
    assert len(result) >= 1
    # All results should have quality_score in metadata
    for sample in result:
        assert "quality_score" in sample.metadata
    print("✓ QualityFilter operator test passed")


def test_registry():
    """Test operator registry."""
    from alignjuice.core.registry import Registry

    # Check that operators are registered
    operators = Registry.list_operators()
    assert "exact_dedup" in operators
    assert "semantic_dedup" in operators
    assert "quality_filter" in operators

    # Check that we can get operator class
    ExactDedup = Registry.get_operator("exact_dedup")
    assert ExactDedup is not None
    print("✓ Registry test passed")


def test_pipeline_basic():
    """Test basic pipeline functionality."""
    from alignjuice.core.data_container import DataContainer
    from alignjuice.core.pipeline import Pipeline
    from alignjuice.stages.base import BaseStage
    from alignjuice.operators.dedup.exact import ExactDedup

    # Create a simple stage
    class SimpleStage(BaseStage):
        name = "simple_stage"

        def process(self, data: DataContainer) -> DataContainer:
            return self._apply_operators(data)

    # Create pipeline
    pipeline = Pipeline()
    stage = SimpleStage(operators=[ExactDedup(field="instruction")])
    pipeline.add_stage(stage)

    # Create test data
    raw_data = [
        {"id": "1", "instruction": "Test", "input": "", "output": "Response 1", "category": "daily"},
        {"id": "2", "instruction": "Test", "input": "", "output": "Response 2", "category": "daily"},
        {"id": "3", "instruction": "Other", "input": "", "output": "Response 3", "category": "daily"},
    ]
    data = DataContainer.from_list(raw_data)

    # Run pipeline
    result = pipeline.run(data)

    assert len(result.data) == 2  # One duplicate removed
    assert result.elapsed_time > 0
    print("✓ Pipeline basic test passed")


def test_config_schema():
    """Test configuration schema."""
    from alignjuice.config.schema import PipelineConfig, StageConfig, OperatorConfig

    config = PipelineConfig(
        name="test_pipeline",
        version="1.0.0",
        stages=[
            StageConfig(
                name="s1_data_juicer",
                operators=[
                    OperatorConfig(name="exact_dedup", params={"field": "instruction"}),
                    OperatorConfig(name="quality_filter", params={"threshold": 0.8}),
                ],
            )
        ],
        dedup_threshold=0.95,
        quality_threshold=0.8,
    )

    assert config.name == "test_pipeline"
    assert len(config.stages) == 1
    assert len(config.stages[0].operators) == 2
    print("✓ Config schema test passed")


def test_io_operations(tmp_path):
    """Test IO read/write operations."""
    from alignjuice.core.data_container import DataContainer
    from alignjuice.io import read_data, write_data

    # Create test data
    raw_data = [
        {"id": "1", "instruction": "Test 1", "input": "", "output": "Output 1", "category": "daily"},
        {"id": "2", "instruction": "Test 2", "input": "", "output": "Output 2", "category": "factual"},
    ]
    data = DataContainer.from_list(raw_data)

    # Write to JSONL
    output_path = tmp_path / "test_output.jsonl"
    write_data(data, output_path)

    assert output_path.exists()

    # Read back
    loaded_data = read_data(output_path)

    assert len(loaded_data) == 2
    assert loaded_data[0].instruction == "Test 1"
    print("✓ IO operations test passed")


def run_all_tests():
    """Run all minimal tests."""
    import tempfile
    from pathlib import Path

    print("\n" + "=" * 50)
    print("Running AlignJuice Minimal Unit Tests")
    print("=" * 50 + "\n")

    test_data_container_creation()
    test_data_container_from_list()
    test_exact_dedup_operator()
    test_quality_filter_operator()
    test_registry()
    test_pipeline_basic()
    test_config_schema()

    # IO test needs temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_io_operations(Path(tmp_dir))

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_all_tests()
