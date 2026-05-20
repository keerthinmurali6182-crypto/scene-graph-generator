import torch
import torch.nn as nn

class SceneGraphExtractionNetwork(nn.Module):
    def __init__(self, d_in=512, num_objs=20, num_attrs=15, num_preds=10):
        super(SceneGraphExtractionNetwork, self).__init__()
        
        # Head 1: Object Classification Network Layer
        self.object_classifier = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, num_objs)
        )
        
        # Head 2: Characteristic Attribute Predictor Layer (Multi-label Output)
        self.attribute_classifier = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, num_attrs),
            nn.Sigmoid() # Sigmoid for independent multi-label features
        )
        
        # Head 3: Dual-Node Edge Relationship Classifier 
        # Ingests concatenated feature bounds from subject and target object pairs
        self.relationship_classifier = nn.Sequential(
            nn.Linear(d_in * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_preds)
        )

    def forward(self, object_features, compute_relationships=False):
        # Predict properties across detached spatial proposal maps
        obj_logits = self.object_classifier(object_features)
        attr_probabilities = self.attribute_classifier(object_features)
        
        rel_logits = None
        if compute_relationships and object_features.size(0) >= 2:
            # Pairwise feature aggregation to model directed edge semantics
            sub_feats = object_features[0].unsqueeze(0) # Node A features
            obj_feats = object_features[1].unsqueeze(0) # Node B features
            paired_features = torch.cat([sub_feats, obj_feats], dim=1)
            rel_logits = self.relationship_classifier(paired_features)
            
        return obj_logits, attr_probabilities, rel_logits