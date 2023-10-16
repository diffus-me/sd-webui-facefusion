from typing import Optional, Tuple, List
import tempfile
import gradio

import facefusion.choices
import facefusion.globals
from facefusion import wording
from facefusion.typing import OutputVideoEncoder
from facefusion.utilities import is_image, is_video
from facefusion.uis.typing import Update, ComponentName
from facefusion.uis.core import get_ui_component, register_ui_component

OUTPUT_PATH_TEXTBOX : Optional[gradio.Textbox] = None
OUTPUT_IMAGE_QUALITY_SLIDER : Optional[gradio.Slider] = None
OUTPUT_VIDEO_ENCODER_DROPDOWN : Optional[gradio.Dropdown] = None
OUTPUT_VIDEO_QUALITY_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global OUTPUT_PATH_TEXTBOX
	global OUTPUT_IMAGE_QUALITY_SLIDER
	global OUTPUT_VIDEO_ENCODER_DROPDOWN
	global OUTPUT_VIDEO_QUALITY_SLIDER

	OUTPUT_PATH_TEXTBOX = gradio.Textbox(
		label = wording.get('output_path_textbox_label'),
		value = facefusion.globals.output_path or tempfile.gettempdir(),
		max_lines = 1,
		visible = False,
	)
	OUTPUT_IMAGE_QUALITY_SLIDER = gradio.Slider(
		label = wording.get('output_image_quality_slider_label'),
		value = facefusion.globals.output_image_quality,
		step = 1,
		minimum = 0,
		maximum = 100,
		# visible = is_image(facefusion.globals.target_path)
		visible = False,
	)
	OUTPUT_VIDEO_ENCODER_DROPDOWN = gradio.Dropdown(
		label = wording.get('output_video_encoder_dropdown_label'),
		choices = facefusion.choices.output_video_encoders,
		value = facefusion.globals.output_video_encoder,
		# visible = is_video(facefusion.globals.target_path)
		visible = False
	)
	OUTPUT_VIDEO_QUALITY_SLIDER = gradio.Slider(
		label = wording.get('output_video_quality_slider_label'),
		value = facefusion.globals.output_video_quality,
		step = 1,
		minimum = 0,
		maximum = 100,
		# visible = is_video(facefusion.globals.target_path)
		visible = False
	)
	register_ui_component('output_path_textbox', OUTPUT_PATH_TEXTBOX)
	register_ui_component('output_image_quality_slider', OUTPUT_IMAGE_QUALITY_SLIDER)
	register_ui_component('output_video_encoder_dropdown', OUTPUT_VIDEO_ENCODER_DROPDOWN)
	register_ui_component('output_video_quality_slider', OUTPUT_VIDEO_QUALITY_SLIDER)


# def listen() -> None:
	# OUTPUT_PATH_TEXTBOX.change(update_output_path, inputs = OUTPUT_PATH_TEXTBOX)
	# OUTPUT_IMAGE_QUALITY_SLIDER.change(update_output_image_quality, inputs = OUTPUT_IMAGE_QUALITY_SLIDER)
	# OUTPUT_VIDEO_ENCODER_DROPDOWN.select(update_output_video_encoder, inputs = OUTPUT_VIDEO_ENCODER_DROPDOWN)
	# OUTPUT_VIDEO_QUALITY_SLIDER.change(update_output_video_quality, inputs = OUTPUT_VIDEO_QUALITY_SLIDER)
	# multi_component_names : List[ComponentName] =\
	# [
	# 	'source_image',
	# 	'target_image',
	# 	'target_video'
	# ]
	# for component_name in multi_component_names:
	# 	component = get_ui_component(component_name)
	# 	if component:
	# 		for method in [ 'upload', 'change', 'clear' ]:
	# 			getattr(component, method)(remote_update, outputs = [ OUTPUT_IMAGE_QUALITY_SLIDER, OUTPUT_VIDEO_ENCODER_DROPDOWN, OUTPUT_VIDEO_QUALITY_SLIDER ])

def listen() -> None:
	target_image = get_ui_component("target_image")
	if target_image:
		for method in [ 'upload', 'change', 'clear' ]:
			getattr(target_image, method)(
				remote_update_image,
				inputs = target_image,
				outputs = [ OUTPUT_IMAGE_QUALITY_SLIDER, OUTPUT_VIDEO_ENCODER_DROPDOWN, OUTPUT_VIDEO_QUALITY_SLIDER ]
			)
	target_video = get_ui_component("target_video")
	if target_video:
		for method in [ 'upload', 'change', 'clear' ]:
			getattr(target_video, method)(
				remote_update_video,
				inputs = target_video,
				outputs = [ OUTPUT_IMAGE_QUALITY_SLIDER, OUTPUT_VIDEO_ENCODER_DROPDOWN, OUTPUT_VIDEO_QUALITY_SLIDER ]
			)


def remote_update_image(target_image) -> Tuple[Update, Update, Update]:
	if target_image is not None:
		return gradio.update(visible = True), gradio.update(visible = False), gradio.update(visible = False)
	return gradio.update(visible = False), gradio.update(visible = False), gradio.update(visible = False)

def remote_update_video(target_video) -> Tuple[Update, Update, Update]:
	if target_video is not None:
		return gradio.update(visible = False), gradio.update(visible = True), gradio.update(visible = True)
	return gradio.update(visible = False), gradio.update(visible = False), gradio.update(visible = False)


def remote_update() -> Tuple[Update, Update, Update]:
	if is_image(facefusion.globals.target_path):
		return gradio.update(visible = True), gradio.update(visible = False), gradio.update(visible = False)
	if is_video(facefusion.globals.target_path):
		return gradio.update(visible = False), gradio.update(visible = True), gradio.update(visible = True)
	return gradio.update(visible = False), gradio.update(visible = False), gradio.update(visible = False)


def update_output_path(output_path : str) -> None:
	facefusion.globals.output_path = output_path


def update_output_image_quality(output_image_quality : int) -> None:
	facefusion.globals.output_image_quality = output_image_quality


def update_output_video_encoder(output_video_encoder: OutputVideoEncoder) -> None:
	facefusion.globals.output_video_encoder = output_video_encoder


def update_output_video_quality(output_video_quality : int) -> None:
	facefusion.globals.output_video_quality = output_video_quality
