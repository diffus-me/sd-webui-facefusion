import cv2
import gradio

from facefusion.typing import Frame
from facefusion.uis.core import get_ui_component, register_ui_component
from facefusion.uis.typing import Update

ID_TASK: gradio.Label
WIDTH: gradio.Number
HEIGHT: gradio.Number


def render() -> None:
    global ID_TASK
    global WIDTH
    global HEIGHT

    ID_TASK = gradio.Label(visible=False)
    WIDTH = gradio.Number(visible=False)
    HEIGHT = gradio.Number(visible=False)

    register_ui_component("id_task", ID_TASK)
    register_ui_component("width", WIDTH)
    register_ui_component("height", HEIGHT)


def listen() -> None:
    global WIDTH
    global HEIGHT

    target_video = get_ui_component("target_video")
    for method in ["upload", "change", "clear"]:
        getattr(target_video, method)(
            _update_video_resolution, inputs=[target_video], outputs=[WIDTH, HEIGHT]
        )

    target_image = get_ui_component("target_image")
    for method in ["upload", "change", "clear"]:
        getattr(target_image, method)(
            _update_image_resolution, inputs=[target_image], outputs=[WIDTH, HEIGHT]
        )

    resolution_upater = (
        f"monitorThisParam('facefusion_interface', 'extensions.facefusion', 'width')",
    )
    WIDTH.change(None, inputs=[], outputs=[WIDTH], _js=resolution_upater)
    HEIGHT.change(None, inputs=[], outputs=[HEIGHT], _js=resolution_upater)

    n_iter_updater = (
        f"monitorThisParam('facefusion_interface', 'extensions.facefusion', 'n_iter', extractor = (x) => Math.max((x[1] - x[0]), 1) * x[2].length)",
    )
    frame_processors_checkbox_group = get_ui_component("frame_processors_checkbox_group")
    trim_frame_start_slider = get_ui_component("trim_frame_start_slider")
    trim_frame_end_slider = get_ui_component("trim_frame_end_slider")

    frame_processors_checkbox_group.change(
        None,
        inputs=[trim_frame_start_slider, trim_frame_end_slider, frame_processors_checkbox_group],
        outputs=[],
        _js=n_iter_updater,
    )
    trim_frame_start_slider.change(
        None,
        inputs=[trim_frame_start_slider, trim_frame_end_slider, frame_processors_checkbox_group],
        outputs=[],
        _js=n_iter_updater,
    )
    trim_frame_end_slider.change(
        None,
        inputs=[trim_frame_start_slider, trim_frame_end_slider, frame_processors_checkbox_group],
        outputs=[],
        _js=n_iter_updater,
    )


def _update_video_resolution(video_path: str) -> tuple[Update, Update]:
    if video_path is not None:
        capture = cv2.VideoCapture(video_path)
        assert capture.isOpened()
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()

        return gradio.update(value=width), gradio.update(value=height)

    return gradio.update(value=None), gradio.update(value=None)


def _update_image_resolution(image: Frame | None) -> tuple[Update, Update]:
    if image is not None:
        shape = image.shape
        return gradio.update(value=shape[1]), gradio.update(value=shape[0])

    return gradio.update(value=None), gradio.update(value=None)
