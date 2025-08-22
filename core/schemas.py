"""
Core data schemas for layout clustering project.
"""

from typing import List, Literal

from pydantic import BaseModel


class Element(BaseModel):
    """Represents a single design element in a layout."""

    element_class: str
    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        """Calculate element area."""
        return self.width * self.height


class Layout(BaseModel):
    """Represents a complete layout with multiple elements."""

    id: str
    width: int
    height: int
    group_id: str
    elements: List[Element]
    quality: Literal["pass", "fail"]

    @property
    def total_area(self) -> float:
        """Calculate total area of all elements."""
        return sum(elem.area for elem in self.elements)

    @property
    def canvas_area(self) -> float:
        """Calculate total canvas area."""
        return self.width * self.height


def load_layouts_from_json(json_path: str) -> List[Layout]:
    """Load layouts from the assignment JSON file."""
    import json

    with open(json_path) as f:
        data = json.load(f)

    layouts: List[Layout] = []
    for item in data:
        # Convert elements
        converted_elements = []
        for elem in item["elements"]:
            converted_elem = Element(
                element_class=elem["class"], x=elem["x"], y=elem["y"], width=elem["width"], height=elem["height"]
            )
            converted_elements.append(converted_elem)

        # Create layout
        layout = Layout(
            id=item.get("id", f"layout_{len(layouts):03d}"),
            width=item["width"],
            height=item["height"],
            group_id=item["groupId"],
            elements=converted_elements,
            quality=item["quality"],
        )
        layouts.append(layout)

    return layouts
