import os
import yaml
import torch
from data_pipeline import MockVisionFeaturePipeline
from engine_models import SceneGraphExtractionNetwork
from structures import SceneGraphNode, SceneGraphEdge

class SceneGraphGenerationEngine:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "../config/pipeline_config.yaml")
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        cfg = self.config['architecture']
        self.network = SceneGraphExtractionNetwork(
            d_in=cfg['backbone_dim'], 
            num_objs=cfg['num_classes'], 
            num_attrs=cfg['num_attributes'], 
            num_preds=cfg['num_predicates']
        )
        self.network.eval() # Fix architecture to inference evaluation mode
        self.data_pipe = MockVisionFeaturePipeline(self.config)

    def pipeline_run(self) -> dict:
        # 1. Gather feature projections and coordinate metadata from the pipeline
        feats, bboxes, classes, attrs, predicates = self.data_pipe.extract_proposal_features()
        
        # 2. Process feed-forward outputs over target components
        with torch.no_grad():
            obj_logits, attr_probs, rel_logits = self.network(feats, compute_relationships=True)
            
        # Extract highest probability indexes
        detected_obj_idxs = torch.argmax(obj_logits, dim=1).tolist()
        
        graph_nodes = []
        # 3. Assemble Structural Scene Graph Nodes
        for idx in range(len(bboxes)):
            class_str = classes.get(detected_obj_idxs[idx], "unknown_object")
            
            # Isolate attributes crossing the target operational threshold values
            active_attr_idxs = (attr_probs[idx] > self.config['inference']['confidence_threshold']).nonzero(as_tuple=True)[0].tolist()
            matched_attrs = [attrs.get(a, "unknown_property") for a in active_attr_idxs]
            
            node = SceneGraphNode(node_id=idx, label=class_str, bbox=bboxes[idx], attributes=matched_attrs)
            graph_nodes.append(node.to_dict())
            
        # 4. Assemble Structural Scene Graph Edges
        graph_edges = []
        if rel_logits is not None:
            pred_idx = torch.argmax(rel_logits, dim=1).item()
            pred_str = predicates.get(pred_idx, "interacting_with")
            
            # Map structural edge directional tracking
            edge = SceneGraphEdge(subject_id=0, predicate=pred_str, object_id=1)
            graph_edges.append(edge.to_dict())
            
        return {
            "document_format": "Scene_Graph_Representation_Structure",
            "nodes": graph_nodes,
            "edges": graph_edges
        }

if __name__ == "__main__":
    print("🚀 Running Scene Graph Extraction Inference Pipeline Engine...")
    engine = SceneGraphGenerationEngine()
    scene_graph_output = engine.pipeline_run()
    
    # Print clean formatted structural JSON graph outputs
    import json
    print(json.dumps(scene_graph_output, indent=4))