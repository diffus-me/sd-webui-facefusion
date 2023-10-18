#!/usr/bin/env python3

from pathlib import Path

import requests

MODEL_URLS = [
    "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/codeformer.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/GFPGANv1.2.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/GFPGANv1.3.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/GFPGANv1.4.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/GPEN-BFR-512.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/RealESRGAN_x2plus.pth",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/RealESRGAN_x4plus.pth",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/RealESRNet_x4plus.pth",
]

MODELS_DIR = Path(__file__).absolute().parent / ".assets" / "models"


def download_file(url: str, path: Path, chunk_size: int = 500 * 1024 * 1024) -> None:
    response = requests.get(url, stream=True, timeout=10)
    with path.open("wb") as fp:
        for data in response.iter_content(chunk_size=chunk_size):
            fp.write(data)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for url in MODEL_URLS:
        filename = url.rsplit("/", 1)[-1]
        path = MODELS_DIR / filename
        if path.exists():
            print(f"Model '{filename}' exists, skip downloading")
            continue

        print(f"Downloading model '{filename}'")
        download_file(url, path)


if __name__ == "__main__":
    main()
