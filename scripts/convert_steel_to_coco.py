# coding: utf-8


import argparse
import json
import logging
import os
import pprint
import shutil
from pathlib import Path

import tqdm as TQDM
from PIL import Image

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert steel data to coco dataset format"
    )
    parser.add_argument(
        "-i", "--input_path", type=str, required=True, help="Input path"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True, help="Output path"
    )
    parser.add_argument(
        "--image_output_path", type=str, default=None, help="Image output path"
    )
    parser.add_argument("--tqdm", action="store_true", help="Show progress bar")
    parser.add_argument("--copy_image", action="store_true", help="Copy image")

    args = parser.parse_args()

    return args


def convert_steel_to_coco(
    input_path: os.PathLike,
    output_path: os.PathLike,
    image_output_path: os.PathLike,
    tqdm: bool = False,
    copy_image: bool = False,
    **kwds,
):
    images = list()
    annotations = list()
    categories = list()

    categories.append({"id": 0, "name": "number"})

    for folder_path in (
        TQDM.tqdm([p for p in Path(input_path).iterdir()])
        if tqdm
        else Path(input_path).iterdir()
    ):
        category_id = len(categories)
        # Add category
        categories.append(
            {
                "id": len(categories),
                "name": folder_path.stem,
            }
        )

        for image_path in (
            TQDM.tqdm([p for p in folder_path.iterdir()], leave=False)
            if tqdm
            else folder_path.iterdir()
        ):
            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # Add image
            with Image.open(image_path) as image:
                if copy_image and Path(image_output_path).is_dir():
                    shutil.copy(image_path, Path(image_output_path, image_path.name))

                image_id = len(images)
                images.append(
                    {
                        "file_name": image_path.name,
                        "width": image.width,
                        "height": image.height,
                        "id": image_id,
                    }
                )

                with Path(image_path.parent, f"{image_path.stem}.txt").open(
                    "r", encoding="utf-8"
                ) as f:
                    for label in json.load(f):
                        (x1, y1, x2, y2) = label["position"]
                        annotations.append(
                            {
                                "image_id": image_id,
                                "iscrowd": 0,
                                "category_id": (
                                    category_id if label["label"] == "steel" else 0
                                ),
                                "area": int((x2 - x1) * (y2 - y1)),
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "id": len(annotations),
                            }
                        )

    with Path(output_path).open("w", encoding="utf-8") as f:
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
    args_dict = vars(args)
    logger.info(pprint.pformat(args_dict))

    convert_steel_to_coco(**args_dict)
