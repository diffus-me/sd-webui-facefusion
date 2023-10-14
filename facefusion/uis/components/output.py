from typing import Tuple, Optional
import gradio

import facefusion.globals
from facefusion import wording
from facefusion.core import limit_resources, conditional_process
from facefusion.uis.core import get_ui_component
from facefusion.utilities import is_image, is_video, normalize_output_path, clear_temp
from facefusion.uis.typing import Update
from modules.system_monitor import monitor_call_context

OUTPUT_IMAGE : Optional[gradio.Image] = None
OUTPUT_VIDEO : Optional[gradio.Video] = None
OUTPUT_START_BUTTON : Optional[gradio.Button] = None
OUTPUT_CLEAR_BUTTON : Optional[gradio.Button] = None


def render() -> None:
	global OUTPUT_IMAGE
	global OUTPUT_VIDEO
	global OUTPUT_START_BUTTON
	global OUTPUT_CLEAR_BUTTON

	OUTPUT_IMAGE = gradio.Image(
		label = wording.get('output_image_or_video_label'),
		interactive = False,
		visible = False
	)
	OUTPUT_VIDEO = gradio.Video(
		label = wording.get('output_image_or_video_label'),
		interactive = False,
	)
	OUTPUT_START_BUTTON = gradio.Button(
		value = wording.get('start_button_label'),
		elem_id = "facefusion_start_button",
		variant = 'primary',
		size = 'sm'
	)
	OUTPUT_CLEAR_BUTTON = gradio.Button(
		value = wording.get('clear_button_label'),
		size = 'sm'
	)


def listen() -> None:
	id_task = get_ui_component('id_task')
	width = get_ui_component('width')
	height = get_ui_component('height')
	output_path_textbox = get_ui_component('output_path_textbox')
	if output_path_textbox:
		OUTPUT_START_BUTTON.click(
			start,
			_js="submit_facefusion_task",
			inputs = [id_task, width, height, output_path_textbox],
			outputs = [ OUTPUT_IMAGE, OUTPUT_VIDEO ]
		)
	OUTPUT_CLEAR_BUTTON.click(clear, outputs = [ OUTPUT_IMAGE, OUTPUT_VIDEO ])


def start(request: gradio.Request, id_task: str, width: int, height: int, output_path : str) -> Tuple[Update, Update]:
	facefusion.globals.output_path = normalize_output_path(facefusion.globals.source_path, facefusion.globals.target_path, output_path)
	limit_resources()
	with monitor_call_context(
		request,
		"extensions.facefusion",
		"extensions.facefusion",
		id_task.removeprefix("task(").removesuffix(")"),
		decoded_params={
			"width": width,
			"height": height,
			"n_iter": 1,
		},
		is_intermediate=False,
	):
		conditional_process(request, width, height)
		if is_image(facefusion.globals.output_path):
			return gradio.update(value = facefusion.globals.output_path, visible = True), gradio.update(value = None, visible = False)
		if is_video(facefusion.globals.output_path):
			return gradio.update(value = None, visible = False), gradio.update(value = facefusion.globals.output_path, visible = True)
		return gradio.update(), gradio.update()


def clear() -> Tuple[Update, Update]:
	if facefusion.globals.target_path:
		clear_temp(facefusion.globals.target_path)
	return gradio.update(value = None), gradio.update(value = None)
