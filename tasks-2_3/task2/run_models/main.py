
import json
from classes import * 
import sys
from vanilla_regression_models import RegressionModelHandler
from vanilla_classification_models import ClassificationModelHandler
from hybrid_task3_models import HybridTask3ModelHandler
from yolo_model import YoloCountingModelHandler
from hybrid_task1_models import HybridTask1ModelHandler

BATCH_SIZE = 16
DEFAULT_MODEL = "resnet50_normal_regression"

corner_model_path = 'models/modelo.pth' # path to the corner detection model for task3_warping

available_models = {
    # vanilla
    "resnet50_normal_classification": "models/classification_resnet50_unbalanced.pth",
    "resnet50_normal_regression": "models/regression_resnet50_unbalanced.pth",
    "resnet50_balanced_classification": "models/classification_resnet50_balanced.pth",
    "resnet50_balanced_regression": "models/regression_resnet50_balanced.pth",

    # hybrid
    "resnet50_hybrid_task1_warping_classification": "models/altered_classification_resnet50_unbalanced.pth",
    "resnet50_hybrid_task1_warping_regression": "models/altered_regression_resnet50_unbalanced.pth",
    "resnet50_hybrid_task3_warping_classification": "models/corners_altered_classification_resnet50_unbalanced.pth",
    "resnet50_hybrid_task3_warping_regression": "models/corners_altered_regression_resnet50_unbalanced.pth",

    #yolo
    "yolo_counting": "runs/detect/chess_piece_detection/weights/best.pt",
}

def run_task2_model(input_path="input.json", model_name="resnet50_normal_regression", output_path="output.json"):
    handler = get_model_handler(model_name)

    #import image_paths
    data = json.load(open(input_path))
    image_paths = data["image_files"]

    dataset = ImageDataset(image_paths, handler.preprocess())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    predictions = []
    for batch_imgs, batch_paths in dataloader:
        preds = handler.predict_batch(batch_imgs)
        for path, pred in zip(batch_paths, preds):
            predictions.append({
                "image": path,
                "num_pieces": pred
            })

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)

def get_model_handler(model_key):
    if "yolo_counting" in model_key:
        return YoloCountingModelHandler(available_models[model_key])
    elif "hybrid_task3" in model_key:
        return HybridTask3ModelHandler(available_models[model_key], corner_model_path=corner_model_path)
    elif "hybrid_task1" in model_key:
        return HybridTask1ModelHandler(available_models[model_key])
    elif "regression" in model_key:
        return RegressionModelHandler(available_models[model_key])
    elif "classification" in model_key:
        return ClassificationModelHandler(available_models[model_key])
    else:
        raise ValueError(f"Unsupported model type for key: {model_key}")
    
if __name__ == "__main__":
    input_json = sys.argv[1] if len(sys.argv) > 1 else "input.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.json"
    model_name = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL

    run_task2_model(input_path=input_json, model_name=model_name, output_path=output_path)
   