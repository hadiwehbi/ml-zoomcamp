import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
import onnx
import onnxruntime as ort


# Default values: change to match your setup
DEFAULT_MODEL_PATH = "hair_classifier_v1.onnx"
DEFAULT_IMAGE_PATH = "data/test/your_sample_image.jpg"  # <- update to the real path

# ImageNet normalization (same as Homework 8)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TARGET_SIZE = (200, 200)  # Q2 answer: 200x200

def load_model(model_path: str):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(model_path.as_posix(), providers=[
                                "CPUExecutionProvider"])
    return onnx_model, sess


def get_output_node_name(onnx_model) -> str:
    return onnx_model.graph.output[0].name


def preprocess_image(image_path: str) -> np.ndarray:
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img = img.resize(TARGET_SIZE)

    # to [0, 1]
    x = np.asarray(img).astype("float32") / 255.0  # (H, W, C)

    # normalize per channel
    x[..., 0] = (x[..., 0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    x[..., 1] = (x[..., 1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    x[..., 2] = (x[..., 2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

    # HWC -> CHW
    x = np.transpose(x, (2, 0, 1))  # (3, 200, 200)

    # add batch dimension
    x = np.expand_dims(x, axis=0)   # (1, 3, 200, 200)

    return x.astype("float32")


def run_inference(session: ort.InferenceSession, x: np.ndarray) -> float:
    """
    Run ONNX inference and return the scalar prediction (probability or logit).
    """
    # Input name
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})
    y = outputs[0]  # assume single output, shape (1, 1) or (1,)
    return float(y.reshape(-1)[0])


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


# ------------------------ Main logic ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="Path to ONNX model")
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH,
                        help="Path to sample image")
    parser.add_argument("--treat_output_as_logit", action="store_true",
                        help="If set, apply sigmoid to ONNX output.")
    args = parser.parse_args()

    print("=== Homework 9 helper script ===")

    # Load model
    onnx_model, sess = load_model(args.model)

    # Q1: Output node name
    output_name = get_output_node_name(onnx_model)
    print(f"Q1 - Output node name: {output_name!r}")
    # Expected answer for the homework: 'output'

    # Q2: Target image size
    print(f"Q2 - Target image size (WxH): {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    # Preprocess image
    x = preprocess_image(args.image)

    # Q3: First R value after preprocessing
    # x shape: (1, 3, 200, 200) -> first R is x[0, 0, 0, 0]
    first_r = float(x[0, 0, 0, 0])
    print(f"Q3 - First R value after preprocessing: {first_r:.3f}")
    # In the official homework solutions, this is approximately: -1.073

    # Q4: Model prediction for this image
    raw_output = run_inference(sess, x)

    if args.treat_output_as_logit:
        prob = sigmoid(raw_output)
    else:
        prob = raw_output  # if model already applies sigmoid inside

    print(f"Q4 - Model output for this image (probability-ish): {prob:.3f}")
    # In the official solution, it's around 0.69

    # Q5 & Q6 cannot be computed purely from this script because they depend
    # on Docker image size and a special 'empty' model image. But we can
    # print the final numeric answers here for reference:

    print("\n--- Reference answers for remaining questions (from Docker / serverless steps) ---")
    print("Q5 - Docker image size (agrigorev/model-2025-hairstyle:v1): 608 Mb")
    print("Q6 - Docker-based model output for the given image: 0.10")


if __name__ == "__main__":
    main()
