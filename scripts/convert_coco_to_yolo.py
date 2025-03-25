# coding: utf-8


import argparse
import json
import logging
import os
import pprint
import shutil
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
        "--label_minus_one",
        action="store_true",
        help="Label number - 1, convert 1-indexes to 0-indexes, like coco dataset format start index is 1 not 0",
    )
    parser.add_argument("--tqdm", action="store_true", help="Show progress bar")

    args = parser.parse_args()

    return args


def xywh2cxcywh(
    x: float,
    y: float,
    w: float,
    h: float,
    width: float = None,
    height: float = None,
) -> tuple[float, float, float, float]:
    cx = x + w / 2
    cy = y + h / 2
    if width and height:
        cx = cx / width
        cy = cy / height
        w = w / width
        h = h / height
    return (cx, cy, w, h)


def convert_coco_to_yolo(
    input_annotation_path: os.PathLike,
    input_image_path: os.PathLike,
    output_path: os.PathLike,
    label_minus_one: bool = False,
    tqdm: bool = False,
    **kwds,
):
    Path(output_path, "labels").mkdir(parents=True, exist_ok=True)
    if input_image_path and Path(input_image_path).is_dir():
        Path(output_path, "images").mkdir(exist_ok=True)

    with Path(input_annotation_path).open("r", encoding="utf-8") as f:
        annotation_data = json.load(f)

    # Pre process image id to image info
    images: dict[str, dict[str, str]] = dict()
    for image in annotation_data.get("images", []):
        images[image["id"]] = image

    # Pre process annotation
    annotations: dict[str, list[dict[str, str]]] = dict()
    for annotation in annotation_data.get("annotations", []):
        if annotation["image_id"] not in annotations:
            annotations[annotation["image_id"]] = list()

        annotations[annotation["image_id"]].append(annotation)

    # Save classes.txt
    with (Path(output_path).parent / "classes.txt").open("w", encoding="utf-8") as f:
        for category in annotation_data.get("categories", []):
            f.write(category["name"] + "\n")

    # Process annotation
    iter_data = [item for item in annotations.items()]
    for image_id, _annotations in TQDM.tqdm(iter_data) if tqdm else iter_data:
        with Path(
            output_path,
            "labels",
            f"{Path(images[image_id]['file_name']).stem}.txt",
        ).open("w", encoding="utf-8") as f:
            for annotation in _annotations:
                f.write(
                    " ".join(
                        [
                            str(
                                annotation["category_id"] - 1
                                if label_minus_one
                                else annotation["category_id"]
                            ),
                            *list(
                                str(p)
                                for p in xywh2cxcywh(
                                    *annotation["bbox"],
                                    width=images[image_id]["width"],
                                    height=images[image_id]["height"],
                                )
                            ),
                        ]
                    )
                    + "\n"
                )

        if input_image_path and Path(input_image_path).is_dir():
            shutil.copy(
                Path(input_image_path, Path(images[image_id]["file_name"]).name),
                Path(output_path, "images", Path(images[image_id]["file_name"]).name),
            )


if __name__ == "__main__":
    args = arg_parser()
    args_dict = vars(args)
    logger.info(pprint.pformat(args_dict))

    convert_coco_to_yolo(**args_dict)
