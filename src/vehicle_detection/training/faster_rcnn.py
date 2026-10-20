import os

import torch
import torchvision
import yaml
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def load_config(config_path=None):
    """Loads the Faster R-CNN configuration using dynamic absolute paths."""
    if config_path is None:
        # Gets the directory where this python file lives
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigates up 3 folders to the project root, then into configs/train/
        config_path = os.path.join(current_dir, "../../../configs/train/faster_rcnn.yaml")
        
    with open(config_path) as file:
        return yaml.safe_load(file)

def setup_faster_rcnn(num_classes):
    """
    Sets up the classic two-stage Faster R-CNN baseline.
    """
    print("--- Initializing Faster R-CNN Baseline ---")
    # Load ResNet-50 backbone pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Replace the classification head to fit our categories (+1 for background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    return model

if __name__ == "__main__":
    config = load_config()
    num_classes = len(config["data"]["classes"])
    
    model = setup_faster_rcnn(num_classes)
    print(f"Faster R-CNN initialized with {num_classes} classes.")
