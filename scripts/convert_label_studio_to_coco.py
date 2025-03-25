# coding: utf-8

import argparse
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import tqdm as TQDM

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert coco dataset to YOLO dataset format"
    )
    parser.add_argument(
        "-ia",
        "--input_annotation_path",
        type=str,
        required=True,
        help="Input annotation path",
    )
    parser.add_argument(
        "-ii",
        "--input_image_path",
        type=str,
        default=None,
        help="Input image path",
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True, help="Output path"
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=None,
        metavar="[0-1]",
        help="Train data size",
    )
    parser.add_argument("--tqdm", action="store_true", help="Show progress bar")

    args = parser.parse_args()

    return args


def _get_image_folder_name(
    image_path: os.PathLike,
    image_name: os.PathLike,
) -> str:
    for folder in Path(image_path).iterdir():
        filenames = [
            file.name
            for file in folder.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
        if Path(image_name).name in filenames:
            return folder.name
    return None


def split_coco_train_and_val(
    images: list[dict],
    annotations: list[dict],
    categories: list[dict],
    train_size: float,
) -> tuple[
    tuple[list[dict], list[dict], list[dict]], tuple[list[dict], list[dict], list[dict]]
]:
    """
    Split COCO dataset into training and validation sets while maintaining category proportions.

    Args:
        images (list[dict]): list of image metadata.
        annotations (list[dict]): list of annotations.
        categories (list[dict]): list of categories.
        train_size (float): Proportion of each category to use for training (e.g., 0.8 for 80%).

    Returns:
        tuple: ((train_images, train_annotations, categories), (val_images, val_annotations, categories))
    """
    # Group images by category
    category_to_images = defaultdict(set)

    for anno in annotations:
        if anno["category_id"] not in [0]:
            category_to_images[anno["category_id"]].add(anno["image_id"])

    train_image_ids = set()
    val_image_ids = set()

    # Split each category individually
    for category_id, image_ids in category_to_images.items():
        image_ids = list(image_ids)
        random.shuffle(image_ids)
        train_count = int(len(image_ids) * train_size)
        train_image_ids.update(image_ids[:train_count])
        val_image_ids.update(image_ids[train_count:])

    # Create train and val image lists
    train_images = [img for img in images if img["id"] in train_image_ids]
    val_images = [img for img in images if img["id"] in val_image_ids]

    # Create train and val annotation lists
    train_annotations = [
        anno for anno in annotations if anno["image_id"] in train_image_ids
    ]
    val_annotations = [
        anno for anno in annotations if anno["image_id"] in val_image_ids
    ]

    return (
        (
            train_images,
            train_annotations,
            categories,
        ),
        (
            val_images,
            val_annotations,
            categories,
        ),
    )


def convert_label_studio_to_coco(
    input_annotation_path: os.PathLike,
    input_image_path: os.PathLike,
    output_path: os.PathLike,
    train_size: float = 0.0,
    tqdm: bool = False,
):
    Path(output_path, "annotations").mkdir(exist_ok=True, parents=True)
    Path(output_path, "images").mkdir(exist_ok=True)

    with Path(input_annotation_path).open(mode="r", encoding="utf-8") as f:
        ground_annotation = json.load(fp=f)

    images = list()
    annotations = list()
    categories = [
        "number",
    ]

    for annotation in TQDM.tqdm(ground_annotation) if tqdm else ground_annotation:
        image_name = Path(annotation["image"].replace("?d=", "")).name
        category_name = _get_image_folder_name(
            image_path=input_image_path,
            image_name=annotation["image"],
        )
        logger.debug(f"category_name: {category_name}")
        logger.debug(f"image_name: {image_name}")
        if category_name not in categories:  # Add category
            categories.append(category_name)
        category_id = categories.index(category_name)

        # Add image
        shutil.copy(
            Path(input_image_path, category_name, image_name),
            Path(output_path, "images", image_name),
        )

        image_id = len(images)
        images.append(
            {
                "file_name": image_name,
                "width": annotation["label"][0]["original_width"],
                "height": annotation["label"][0]["original_height"],
                "id": image_id,
            }
        )

        for label in annotation["label"]:
            (x, y, w, h) = (
                int(label["x"] * label["original_width"] / 100),
                int(label["y"] * label["original_height"] / 100),
                int(label["width"] * label["original_width"] / 100) + 1,
                int(label["height"] * label["original_height"] / 100) + 1,
            )
            annotations.append(
                {
                    "image_id": image_id,
                    "iscrowd": 0,
                    "category_id": category_id
                    if label["rectanglelabels"][0] == "steel"
                    else 0,
                    "area": int(w * h),
                    "bbox": [x, y, w, h],
                    "id": len(annotations),
                }
            )

    categories = [
        {"id": index, "name": category} for index, category in enumerate(categories)
    ]

    if train_size:
        # Split val data
        (
            (
                images,
                annotations,
                categories,
            ),
            (
                val_images,
                val_annotations,
                categories,
            ),
        ) = split_coco_train_and_val(
            images=images,
            annotations=annotations,
            categories=categories,
            train_size=train_size,
        )

        with Path(output_path, "annotations", "val.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "images": val_images,
                    "annotations": val_annotations,
                    "categories": categories,
                },
                f,
            )

    with Path(output_path, "annotations", "train.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
            f,
        )


if __name__ == "__main__":
    args = arg_parser()
    convert_label_studio_to_coco(**vars(args))
