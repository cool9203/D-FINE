# coding: utf-8


import json
import os
import time
from pathlib import Path

import httpx
import tqdm as TQDM
from PIL import Image as PILImage
from PIL import ImageDraw


def calc_iou(
    area_1: tuple[float, float, float, float],
    area_2: tuple[float, float, float, float],
) -> float:
    """Calculate IOU with 2 area

    Args:
        area_1 (tuple[float, float, float, float]): (x1, y1, x2, y2)
        area_2 (tuple[float, float, float, float]): (x1, y1, x2, y2)

    Returns:
        float: iou result, number in 0~1
    """
    # 解析座標
    x1_1, y1_1, x2_1, y2_1 = area_1
    x1_2, y1_2, x2_2, y2_2 = area_2

    # 計算相交區域 (Intersection)
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # 計算各自區域 (Union)
    area_1_size = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_2_size = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area_1_size + area_2_size - inter_area

    # 避免除零錯誤
    if union_area == 0:
        return 0.0

    # 計算 IoU
    return inter_area / union_area


def _detect_one_result(
    url: str,
    threshold: float,
    image_path: os.PathLike,
    save_path: os.PathLike,
    retry: int,
    annotations: dict[str, list[dict[str, int | list[int]]]] = None,
    iou_threshold: float = 0.9,
) -> tuple[bool, bool, bool]:
    detect_number_count = 0
    detect_steel_count = 0
    classification_steel_count = 0

    # Get ground truth
    if annotations and Path(image_path).name not in annotations:
        ValueError(f"{Path(image_path).name} not in ground truth data")
    ground_truth = annotations.get(Path(image_path).name)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(image_path).open(mode="rb") as img:
        for _ in range(retry):
            try:
                resp = httpx.post(
                    url=url,
                    data={
                        "threshold": threshold,
                    },
                    files={"image": img},
                )
                response = resp.json()
                break
            except Exception:
                time.sleep(5)

    image = PILImage.open(image_path)
    image_draw = ImageDraw.Draw(im=image)

    # Clear steel repeat detect
    clear_hight_iou_steel_results = list()
    for i in range(len(response["results"])):
        if response["results"][i]["name"] in ["number"]:
            clear_hight_iou_steel_results.append(response["results"][i])
            continue
        repeat = False
        for j in range(i + 1, len(response["results"])):
            if (
                response["results"][i]["name"] == response["results"][j]["name"]
                and calc_iou(
                    response["results"][i]["bbox"], response["results"][j]["bbox"]
                )
                >= 0.95
            ):
                repeat = True
        if not repeat:
            clear_hight_iou_steel_results.append(response["results"][i])
    response["results"] = clear_hight_iou_steel_results

    names = [
        result["name"]
        for result in response["results"]
        if result["name"] not in ["number"]
    ]
    multi_steel = len(names) > 1

    for result in response["results"]:
        (x1, y1, x2, y2) = result["bbox"]

        classification_steel = False
        for label in ground_truth:
            if (
                result["name"] == label["category_name"]
                and calc_iou(
                    (x1, y1, x2, y2),
                    (
                        label["bbox"][0],
                        label["bbox"][1],
                        label["bbox"][0] + label["bbox"][2],
                        label["bbox"][1] + label["bbox"][3],
                    ),
                )
                >= iou_threshold
            ):
                if result["name"] in ["number"]:
                    detect_number_count += 1
                elif not multi_steel:
                    detect_steel_count += 1

            if result["name"] == label["category_name"] and result["name"] not in [
                "number"
            ]:
                classification_steel = True

        classification_steel_count += 1 if classification_steel else 0

        if result["name"] in ["number"]:
            image_draw.rectangle(((x1, y1), (x2, y2)), outline=(255, 0, 0), width=2)
        elif not multi_steel and result["name"] in str(save_path):
            image_draw.rectangle(((x1, y1), (x2, y2)), outline=(0, 0, 255), width=2)
            image_draw.text(
                (x1, y1),
                text=result["name"],
                fill=(255, 0, 0),
            )
        else:
            image_draw.text(
                (0, 0),
                text=", ".join(set(names)),
                fill=(255, 0, 0),
            )

    image.save(save_path)

    return (
        detect_number_count
        == len(
            [label for label in ground_truth if label["category_name"] in ["number"]]
        ),
        detect_steel_count
        == len(
            [
                label
                for label in ground_truth
                if label["category_name"] not in ["number"]
            ]
        ),
        classification_steel_count > 0,
    )


def load_coco_data(
    dataset_paths: list[os.PathLike],
) -> dict[str, list[dict[str, str | int | list[int]]]]:
    final_annotations: dict[str, list[dict[str, str | int | list[int]]]] = dict()
    _categories = list()
    for dataset_path in dataset_paths:
        with Path(dataset_path).open(mode="r", encoding="utf-8") as f:
            data = json.load(fp=f)
            (images, annotations, categories) = (
                data["images"],
                data["annotations"],
                data["categories"],
            )
            image_id_to_name = {image["id"]: image["file_name"] for image in images}
            category_id_to_name = {
                category["id"]: category["name"] for category in categories
            }
            for annotation in annotations:
                image_name = image_id_to_name[annotation["image_id"]]
                if image_name not in final_annotations:
                    final_annotations[image_name] = list()
                annotation["category_name"] = category_id_to_name[
                    annotation["category_id"]
                ]
                final_annotations[image_name].append(annotation)

            if not _categories:
                _categories = categories
            elif _categories != categories:
                raise ValueError("categories not equal")
    return final_annotations


def run_detect_result(
    url: str,
    threshold: float,
    image_path: os.PathLike,
    output_path: os.PathLike,
    retry: int = 3,
    tqdm: bool = False,
    ground_truth_paths: list[os.PathLike] = None,
    iou_threshold: float = 0.95,
):
    # Load ground truth data, format is coco
    annotations = (
        load_coco_data(dataset_paths=ground_truth_paths)
        if ground_truth_paths
        else dict()
    )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    iter_files = [p for p in Path(image_path).iterdir()]
    all_count = dict(other=0)
    detect_number_correct = dict(other=0)
    detect_steel_correct = dict(other=0)
    classification_steel_correct = dict(other=0)

    for file in TQDM.tqdm(iter_files) if tqdm else iter_files:
        if file.is_dir():
            iter_filenames = [p for p in file.iterdir()]
            _all_count = 0
            _detect_number_correct = 0
            _detect_steel_correct = 0
            _classification_steel_correct = 0
            for filename in (
                TQDM.tqdm(iter_filenames, leave=False) if tqdm else iter_filenames
            ):
                result = _detect_one_result(
                    url=url,
                    threshold=threshold,
                    image_path=filename,
                    save_path=Path(output_path, file.name, filename.name),
                    retry=retry,
                    annotations=annotations,
                    iou_threshold=iou_threshold,
                )
                _all_count += 1
                _detect_number_correct += result[0]
                _detect_steel_correct += result[1]
                _classification_steel_correct += result[2]
            all_count[file.name] = _all_count
            detect_number_correct[file.name] = _detect_number_correct
            detect_steel_correct[file.name] = _detect_steel_correct
            classification_steel_correct[file.name] = _classification_steel_correct
        else:
            result = _detect_one_result(
                url=url,
                threshold=threshold,
                image_path=file,
                save_path=Path(output_path, file.name),
                retry=retry,
                annotations=annotations,
                iou_threshold=iou_threshold,
            )
            all_count["other"] += 1
            detect_number_correct["other"] += result[0]
            detect_steel_correct["other"] += result[1]
            classification_steel_correct["other"] += result[2]

    print("類別\t總正確率\t框數字正確率\t框鋼材正確率\t鋼材辨識正確率")
    for category_name in all_count.keys():
        all_correct = min(
            [
                detect_number_correct[category_name],
                detect_steel_correct[category_name],
                classification_steel_correct[category_name],
            ]
        )
        print(f"{category_name}", end="")
        print(
            f"\t{all_correct}/{all_count[category_name]}",
            end="",
        )
        print(
            f"\t{detect_number_correct[category_name]}/{all_count[category_name]}",
            end="",
        )
        print(
            f"\t{detect_steel_correct[category_name]}/{all_count[category_name]}",
            end="",
        )
        print(
            f"\t{classification_steel_correct[category_name]}/{all_count[category_name]}"
        )


if __name__ == "__main__":
    run_detect_result(
        url="http://10.70.0.128:21356/api/detect_steel",
        threshold=0.5,
        image_path="./data/test_images",
        output_path="./result/20250324-val",
        tqdm=True,
        ground_truth_paths=[
            "./data/custom/annotations/train.json",
            "./data/custom/annotations/val.json",
        ],
    )

    run_detect_result(
        url="http://10.70.0.128:21356/api/detect_steel",
        threshold=0.5,
        image_path="/mnt/c/Users/ychsu/Downloads/沛波標記data/鋼材辨識/20250227_TMPCO_Icon_Samples",
        output_path="./result/20250324-all",
        tqdm=True,
        ground_truth_paths=[
            "./data/custom/annotations/train.json",
            "./data/custom/annotations/val.json",
        ],
    )
