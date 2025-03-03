# coding: utf-8

import argparse
import base64
import io
import time
from pathlib import Path
from typing import Annotated

import gradio as gr
import torch
import torch.nn
import torchvision.transforms as T
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict

from src.core import YAMLConfig

__model: dict[str, torch.nn.Module] = {
    "model": None,
    "name": None,
}

app = FastAPI()

categories = [
    {"id": 0, "name": "number"},
    {"id": 1, "name": "A001-1"},
    {"id": 2, "name": "B101-2"},
    {"id": 3, "name": "B101-2.1"},
    {"id": 4, "name": "B103-3"},
    {"id": 5, "name": "B104-2"},
    {"id": 6, "name": "B104-2.1"},
    {"id": 7, "name": "B104-3"},
    {"id": 8, "name": "B105-1"},
    {"id": 9, "name": "B105-2"},
    {"id": 10, "name": "B105-4"},
    {"id": 11, "name": "B201-3"},
    {"id": 12, "name": "B201-3.1"},
    {"id": 13, "name": "B201-5"},
    {"id": 14, "name": "B203-2"},
    {"id": 15, "name": "B203-3"},
    {"id": 16, "name": "B203-4"},
    {"id": 17, "name": "B203-5"},
    {"id": 18, "name": "B206-3"},
    {"id": 19, "name": "B208-5"},
    {"id": 20, "name": "B209-3"},
    {"id": 21, "name": "B218-2"},
    {"id": 22, "name": "B218-5"},
    {"id": 23, "name": "B221-4"},
    {"id": 24, "name": "B221-4.1"},
    {"id": 25, "name": "B308-3"},
    {"id": 26, "name": "B321-7"},
    {"id": 27, "name": "B409-5"},
    {"id": 28, "name": "B411-5"},
    {"id": 29, "name": "B419-2"},
    {"id": 30, "name": "B419-2.1"},
    {"id": 31, "name": "B419-5"},
    {"id": 32, "name": "B444-5"},
    {"id": 33, "name": "B444-5.1"},
    {"id": 34, "name": "B501-5"},
    {"id": 35, "name": "B502-2"},
    {"id": 36, "name": "B502-2.1"},
    {"id": 37, "name": "B502-2.2"},
    {"id": 38, "name": "B502-5"},
    {"id": 39, "name": "B502-5.1"},
    {"id": 40, "name": "B502-9"},
    {"id": 41, "name": "B618-7"},
    {"id": 42, "name": "C003-1"},
    {"id": 43, "name": "Cknown-2"},
    {"id": 44, "name": "Dknown-3"},
]
_id2name = {category["id"]: category["name"] for category in categories}
_name2id = {category["name"]: category["id"] for category in categories}


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Run test website to test D-Fine")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, help="Checkpoint name or path")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web server host")
    parser.add_argument("--port", type=int, default=7860, help="Web server port")
    parser.add_argument("--device", type=str, default="cpu", help="Run model device")
    parser.add_argument(
        "--example_folder", type=str, default="example", help="Example folder"
    )

    args = parser.parse_args()

    return args


class DetectSteelResponse(BaseModel):
    image: str
    used_time: float
    results: list[dict[str, float | list[int | float] | str]]

    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_model(
    config: str,
    model_name: str = None,
    device: str = "cpu",
) -> torch.nn.Module:
    cfg = YAMLConfig(config, resume=model_name)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if model_name:
        checkpoint = torch.load(model_name, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(device)

    return model


@app.post("/api/detect_steel")
def detect_steel_api(
    image: Annotated[UploadFile, File()],
    threshold: Annotated[float, Form()] = 0.5,
    img_type: Annotated[str, Form()] = "png",
) -> DetectSteelResponse:
    return _detect_steel(
        image=image.file.read(),
        threshold=threshold,
        img_type=img_type,
    )


def detect_steel(
    image: Image.Image,
    threshold: float,
):
    response = _detect_steel(
        image=image,
        threshold=threshold,
    )
    image_base64 = response.image
    image_type = image_base64.split(";")[0].split(":")[1]  # noqa: F841
    image_base64 = image_base64.split(";")[1].split(",")[1]
    return (
        Image.open(io.BytesIO(base64.b64decode(image_base64.encode("utf-8")))),
        response.used_time,
    )


def _detect_steel(
    image: str | bytes | Image.Image,
    threshold: float,
    img_type: str = "png",
) -> DetectSteelResponse:
    # Process all type to PIL.Image
    if isinstance(image, str):
        image = image.encode("utf-8")
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    if isinstance(image, Image.Image):
        image = image.convert("RGB")

    (model, device) = (__model["model"], __model["device"])

    w, h = image.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(image).unsqueeze(0).to(device)

    start_time = time.time()
    output = model(im_data, orig_size)
    end_time = time.time()

    (labels, boxes, scores) = output

    draw = ImageDraw.Draw(image)
    (label, box, score) = (labels[0], boxes[0], scores[0])

    lab = label[score > threshold]
    box = box[score > threshold]
    scrs = score[score > threshold]

    results = list()
    for j, b in enumerate(box):
        draw.rectangle(list(b), outline="red")
        draw.text(
            (b[0], b[1]),
            text=f"{_id2name[lab[j].item()]} {round(scrs[j].item(), 2)}",
            fill="blue",
        )
        results.append(
            {
                "bbox": b.tolist(),
                "id": lab[j].item(),
                "name": _id2name[lab[j].item()],
                "confidence": round(scrs[j].item(), 2),
            }
        )

    with io.BytesIO() as img_io:
        image.save(img_io, format=img_type)
        img_b64_str = base64.b64encode(img_io.getvalue()).decode("utf-8")

        return DetectSteelResponse(
            image=f"data:{img_type};base64,{img_b64_str}",
            used_time=end_time - start_time,
            results=results,
        )


def test_website(
    model_name: str = None,
    config: str = None,
    device: str = "cpu",
    example_folder: str = "examples",
    **kwds,
) -> gr.Blocks:
    if model_name and __model.get("name") is None and config:
        __model["model"] = load_model(
            config=config,
            model_name=model_name,
            device=device,
        )
        __model["name"] = model_name
        __model["device"] = device

    # Gradio 接口定義
    with gr.Blocks(
        title="鋼材偵測測試網站",
        # css="#component-6 { max-height: 85vh; }",
    ) as blocks:
        gr.Markdown("## 鋼材偵測測試網站")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="上傳圖片",
                    type="pil",
                    height="85vh",
                )

            with gr.Column():
                detect_result = gr.Image(
                    label="偵測結果",
                    type="pil",
                    height="85vh",
                )

        submit_button = gr.Button("偵測")

        with gr.Row():
            _model_name = gr.Textbox(
                label="模型名稱或路徑",
                value=__model.get("name", None),
                visible=not model_name,
            )
            threshold = gr.Slider(
                label="Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.01
            )
            time_usage = gr.Textbox(label="Usage time")

        # Constant augments
        _device = gr.Textbox(value=device, visible=False)

        # Examples
        if Path(example_folder).exists():
            example_files = sorted(
                [
                    (str(path.resolve()), path.name)
                    for path in Path(example_folder).iterdir()
                    if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ],
                key=lambda e: e[1],
            )
            gr.Examples(
                examples=[
                    [
                        Image.open(path),
                    ]
                    for path, name in example_files
                ],
                example_labels=[name for path, name in example_files],
                inputs=[
                    image_input,
                    _model_name,
                    _device,
                ],
            )

        submit_button.click(
            detect_steel,
            inputs=[
                image_input,
                threshold,
            ],
            outputs=[
                detect_result,
                time_usage,
            ],
        )
        return blocks


if __name__ == "__main__":
    args = arg_parser()
    args_dict = vars(args)
    blocks = test_website(**args_dict)

    import uvicorn

    app = gr.mount_gradio_app(
        app=app,
        blocks=blocks,
        path="/",
    )
    uvicorn.run(
        app=app,
        host=args_dict["host"],
        port=args_dict["port"],
    )
