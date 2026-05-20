class SceneGraphNode:
    def __init__(self, node_id: int, label: str, bbox: list, attributes: list = None):
        self.node_id = node_id
        self.label = label
        self.bbox = bbox  # Spatial box coordinates [xmin, ymin, xmax, ymax]
        self.attributes = attributes if attributes else []

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "class_label": self.label,
            "bounding_box": self.bbox,
            "detected_attributes": self.attributes
        }

class SceneGraphEdge:
    def __init__(self, subject_id: int, predicate: str, object_id: int):
        self.subject_id = subject_id      # Source Node ID
        self.predicate = predicate        # Directed Edge interaction string (predicate)
        self.object_id = object_id        # Target Node ID

    def to_dict(self) -> dict:
        return {
            "subject": self.subject_id,
            "predicate": self.predicate,
            "object": self.object_id
        }