"""
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show coco eval metrics")
    parser.add_argument(
        "-i", "--input_path", type=str, required=True, help="Input path"
    )

    args = parser.parse_args()

    return args


def show_d_fine_metrics(
    input_path: os.PathLike,
):
    train_loss_values = list()
    train_loss_bbox_values = list()
    metrics = [[] for _ in range(12)]
    metrics_name = [
        "AP @[ IoU=0.50:0.95 |\n area=   all | maxDets=100 ]",
        "AP @[ IoU=0.50      |\n area=   all | maxDets=100 ]",
        "AP @[ IoU=0.75      |\n area=   all | maxDets=100 ]",
        "AP @[ IoU=0.50:0.95 |\n area= small | maxDets=100 ]",
        "AP @[ IoU=0.50:0.95 |\n area=medium | maxDets=100 ]",
        "AP @[ IoU=0.50:0.95 |\n area= large | maxDets=100 ]",
        "AR @[ IoU=0.50:0.95 |\n area=   all | maxDets=  1 ]",
        "AR @[ IoU=0.50:0.95 |\n area=   all | maxDets= 10 ]",
        "AR @[ IoU=0.50:0.95 |\n area=   all | maxDets=100 ]",
        "AR @[ IoU=0.50:0.95 |\n area= small | maxDets=100 ]",
        "AR @[ IoU=0.50:0.95 |\n area=medium | maxDets=100 ]",
        "AR @[ IoU=0.50:0.95 |\n area= large | maxDets=100 ]",
    ]

    with Path(input_path).open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            for i, value in enumerate(data["test_coco_eval_bbox"]):
                metrics[i].append(value)

            train_loss_values.append(data["train_loss"])
            train_loss_bbox_values.append(data["train_loss_bbox"])

    _, axes = plt.subplots(nrows=2, ncols=6, figsize=(5, 3))
    for i in range(len(axes)):
        for j in range(len(axes[i])):
            axes[i][j].plot(metrics[6 * i + j])
            axes[i][j].set_title(metrics_name[6 * i + j])
            axes[i][j].set_xlabel("Epochs")
            axes[i][j].set_ylabel(metrics_name[6 * i + j][:2])

    plt.show()

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    axes[0].plot(train_loss_values)
    axes[0].set_title("train_loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("train_loss")
    axes[1].plot(train_loss_bbox_values)
    axes[1].set_title("train_loss_bbox")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("train_loss_bbox")
    plt.show()


if __name__ == "__main__":
    args = arg_parser()
    show_d_fine_metrics(**vars(args))
