from transformers import AutoModelForCausalLM
import os
from PIL import Image
import argparse
import json

image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}


def get_image_files(directory: str) -> list[str]:
    image_files = []

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file_name)
            if ext in image_exts:
                image_files.append(file_path)
    
    return image_files


def detect(model, img: Image.Image, label: str):
    detected_objects = model.detect(img, label)["objects"]
    if not detected_objects:
        return []

    bounding_boxes = []
    for obj in detected_objects:
        x_min = obj["x_min"]
        y_min = obj["y_min"]
        x_max = obj["x_max"]
        y_max = obj["y_max"]

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        bounding_boxes.append([x_center, y_center, width, height])
    return bounding_boxes

def get_processed_data(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        return {"image_counter": 0, "files": []}

def main():
    parser = argparse.ArgumentParser(
        description="Automatically annotates YOLO dataset using Moondream visual model."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the source directory containing image files.",
    )
    parser.add_argument(
        "--destination",
        type=str,
        required=True,
        help="Path to the directory where the YOLO dataset will be saved.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",  # '+' means one or more arguments
        type=str,
        required=True,
        help='A list of class names (e.g., "person", "car", "tree"). Must provide at least one class.',
    )
    args = parser.parse_args()

    source_directory = args.source
    destination_directory = args.destination
    class_names = args.classes

    if not os.path.isdir(source_directory):
        print(
            f"Error: The specified images path '{source_directory}' is not a valid directory."
        )
        return

    if not os.path.isdir(destination_directory):
        print(
            f"Error: The specified dataset path '{destination_directory}' is not a valid directory."
        )
        return

    images_train_dir = os.path.join(destination_directory, "images", "train")
    labels_train_dir = os.path.join(destination_directory, "labels", "train")

    processed_data_path = os.path.join(destination_directory, "processed.json")
    processed_data = get_processed_data(processed_data_path)    
    image_counter = processed_data["image_counter"]
    processed_files = set(processed_data["files"])

    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)

    image_files = get_image_files(source_directory)
    if len(image_files) < 1:
        print("Image files were not found in specified directory")
        return

    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-04-14",
        trust_remote_code=True,
        device_map={"": "cuda"},
    )

    image_counter = 0
    for image_path in image_files:
        if image_path in processed_files:
            continue

        try:
            img = Image.open(image_path)

            for class_name in class_names:
                detected_objects = detect(model, img, class_name)
                if len(detected_objects) != 0:
                    image_filename = os.path.join(
                        images_train_dir, f"{image_counter}.png"
                    )
                    img.save(image_filename)

                    txt_filename = os.path.join(
                        labels_train_dir, f"{image_counter}.txt"
                    )
                    with open(txt_filename, "w") as f:
                        for obj in detected_objects:
                            f.write(f"0 {obj[0]} {obj[1]} {obj[2]} {obj[3]}\n")

                    break

            image_counter += 1

        except Exception as err:
            print(f"Error: {err}")

        finally:
            processed_data["image_counter"] = image_counter
            processed_data["files"].append(image_path)
            with open(processed_data_path, "w") as f:
                json.dump(processed_data, f, indent=2)


if __name__ == "__main__":
    main()
