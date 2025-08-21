"""
Pytest configuration and fixtures for layout clustering tests.
"""

from typing import List

import pytest

from core.schemas import Element, Layout  # type: ignore


@pytest.fixture
def sample_element() -> Element:
    """Create a sample element for testing."""
    return Element(
        element_class="headline",
        x=100,
        y=50,
        width=300,
        height=50
    )


@pytest.fixture
def sample_elements() -> List[Element]:
    """Create a list of sample elements for testing."""
    return [
        Element(element_class="headline", x=100, y=50, width=300, height=50),
        Element(element_class="body", x=100, y=120, width=250, height=80),
        Element(element_class="image", x=400, y=50, width=200, height=150),
        Element(element_class="button", x=100, y=220, width=120, height=40),
    ]


@pytest.fixture
def sample_layout(sample_elements: List[Element]) -> Layout:
    """Create a sample layout for testing."""
    return Layout(
        id="test_layout_001",
        width=600,
        height=400,
        group_id="test_group",
        elements=sample_elements,
        quality="pass"
    )


@pytest.fixture
def empty_layout() -> Layout:
    """Create an empty layout for testing edge cases."""
    return Layout(
        id="empty_layout",
        width=800,
        height=600,
        group_id="empty_group",
        elements=[],
        quality="fail"
    )


@pytest.fixture
def multiple_layouts(sample_elements: List[Element]) -> List[Layout]:
    """Create multiple layouts for clustering tests."""
    layouts = []
    
    # Create variations of the base layout
    for i in range(5):
        layout = Layout(
            id=f"test_layout_{i:03d}",
            width=600 + i * 50,
            height=400 + i * 30,
            group_id=f"group_{i}",
            elements=sample_elements.copy(),
            quality="pass" if i % 2 == 0 else "fail"
        )
        layouts.append(layout)
    
    return layouts
