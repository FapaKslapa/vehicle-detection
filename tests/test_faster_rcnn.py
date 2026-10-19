import pytest
from src.vehicle_detection.training.faster_rcnn import setup_faster_rcnn, load_config

def test_faster_rcnn_initialization():
    # Load test config or use hardcoded number
    num_classes = 8 
    
    # Initialize model
    model = setup_faster_rcnn(num_classes)
    
    # Verify the model is not None
    assert model is not None
    
    # Verify the output features match classes + 1 (for background)
    out_features = model.roi_heads.box_predictor.cls_score.out_features
    assert out_features == num_classes + 1
