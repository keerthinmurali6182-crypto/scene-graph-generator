import torch

class MockVisionFeaturePipeline:
    def __init__(self, config: dict):
        self.cfg = config['architecture']
        
    def extract_proposal_features(self):
        """Generates mock region proposal tensors and categorical mapping dictionaries."""
        # Simulate bounding boxes for 2 tracked objects in an image frame
        mock_boxes = [[34, 52, 210, 400], [180, 200, 450, 490]]
        
        # Mock feature maps [2 objects, 512 channel dimensions]
        mock_features = torch.randn(2, self.cfg['backbone_dim'])
        
        # Index translation charts
        class_map = {0: "person", 1: "bicycle", 2: "backpack"}
        attr_map = {0: "wearing_red", 1: "metallic", 2: "standing"}
        pred_map = {0: "riding", 1: "holding", 2: "next_to"}
        
        return mock_features, mock_boxes, class_map, attr_map, pred_map