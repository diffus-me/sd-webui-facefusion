from typing import Tuple, Optional
import gradio
import cv2

import facefusion.globals
from facefusion import wording
from facefusion.core import limit_resources, conditional_process
from facefusion.uis.core import get_ui_component
from facefusion.utilities import is_image, is_video, normalize_output_path, clear_temp, get_output_path, create_work_dir
from facefusion.uis.typing import Update
from facefusion.typing import FaceRecognition, Frame, FaceAnalyserAge, FaceAnalyserDirection, FaceAnalyserGender
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
	id_task = get_ui_component("id_task")
	width = get_ui_component("width")
	height = get_ui_component("height")
	preview_frame_slider = get_ui_component("preview_frame_slider")
	source_image = get_ui_component("source_image")
	target_image = get_ui_component("target_image")
	target_video = get_ui_component("target_video")
	face_recognition_dropdown = get_ui_component("face_recognition_dropdown")
	frame_processors_checkbox_group = get_ui_component("frame_processors_checkbox_group")
	face_swapper_model_dropdown = get_ui_component("face_swapper_model_dropdown")
	face_enhancer_model_dropdown = get_ui_component("face_enhancer_model_dropdown")
	frame_enhancer_model_dropdown = get_ui_component("frame_enhancer_model_dropdown")
	reference_face_position_gallery_index = get_ui_component("reference_face_position_gallery_index")
	face_analyser_direction_dropdown = get_ui_component("face_analyser_direction_dropdown")
	face_analyser_age_dropdown = get_ui_component("face_analyser_age_dropdown")
	face_analyser_gender_dropdown = get_ui_component("face_analyser_gender_dropdown")
	reference_face_distance_slider = get_ui_component("reference_face_distance_slider")
	face_enhancer_blend_slider = get_ui_component("face_enhancer_blend_slider")
	frame_enhancer_blend_slider = get_ui_component("frame_enhancer_blend_slider")
	output_image_quality_slider = get_ui_component("output_image_quality_slider")
	temp_frame_format_dropdown = get_ui_component("temp_frame_format_dropdown")
	temp_frame_quality_slider = get_ui_component("temp_frame_quality_slider")
	output_video_encoder_dropdown = get_ui_component("output_video_encoder_dropdown")
	output_video_quality_slider = get_ui_component("output_video_quality_slider")
	trim_frame_start_slider = get_ui_component("trim_frame_start_slider")
	trim_frame_end_slider = get_ui_component("trim_frame_end_slider")
	common_options_checkbox_group = get_ui_component("common_options_checkbox_group")

	OUTPUT_START_BUTTON.click(
		start,
		_js="submit_facefusion_task",
		inputs = [
			id_task,
			width,
			height,
			preview_frame_slider,
			source_image,
			target_image,
			target_video,
			face_recognition_dropdown,
			reference_face_position_gallery_index,
			face_analyser_direction_dropdown,
			face_analyser_age_dropdown,
			face_analyser_gender_dropdown,
			frame_processors_checkbox_group,
			face_swapper_model_dropdown,
			face_enhancer_model_dropdown,
			frame_enhancer_model_dropdown,
			reference_face_distance_slider,
			face_enhancer_blend_slider,
			frame_enhancer_blend_slider,
			output_image_quality_slider,
			temp_frame_format_dropdown,
			temp_frame_quality_slider,
			output_video_encoder_dropdown,
			output_video_quality_slider,
			trim_frame_start_slider,
			trim_frame_end_slider,
			common_options_checkbox_group,
		],
		outputs = [ OUTPUT_IMAGE, OUTPUT_VIDEO ]
	)
	OUTPUT_CLEAR_BUTTON.click(clear, outputs = [ OUTPUT_IMAGE, OUTPUT_VIDEO ])


def start(
	request: gradio.Request,
	id_task: str,
	width: int,
	height: int,
	preview_frame_slider: int,
	source_image: Frame | None,
	target_image: Frame | None,
	target_video: str | None,
	face_recognition_dropdown: FaceRecognition,
	reference_face_position_gallery_index: int,
	face_analyser_direction: FaceAnalyserDirection,
	face_analyser_age: FaceAnalyserAge,
	face_analyser_gender: FaceAnalyserGender,
	frame_processors_checkbox_group: list[str],
	face_swapper_model_dropdown: str,
	face_enhancer_model_dropdown: str,
	frame_enhancer_model_dropdown: str,
	reference_face_distance_slider: float,
	face_enhancer_blend_slider: int,
	frame_enhancer_blend_slider: int,
	output_image_quality_slider: int,
	temp_frame_format_dropdown: str,
	temp_frame_quality_slider: int,
	output_video_encoder_dropdown: str,
	output_video_quality_slider: int,
	trim_frame_start_slider: int,
	trim_frame_end_slider: int,
	common_options_checkbox_group: list[str],
) -> Tuple[Update, Update]:
	if source_image is None or (target_image is None and target_video is None):
		return

	source_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
	if target_image is not None:
		target_image = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)

	task_id = id_task.removeprefix("task(").removesuffix(")")
	# facefusion.globals.output_path = normalize_output_path(facefusion.globals.source_path, facefusion.globals.target_path, output_path)
	if target_image is not None:
		extension = ".jpg"
	else:
		extension = ".mp4"
	create_work_dir(task_id)
	output_path = str(get_output_path(task_id, extension))
	limit_resources()
	with monitor_call_context(
		request,
		"extensions.facefusion",
		"extensions.facefusion",
		task_id,
		decoded_params={
			"width": width,
			"height": height,
			"n_iter": 1,
		},
		is_intermediate=False,
	):
		conditional_process(
			request,
			task_id,
			width,
			height,
			output_path,
			preview_frame_slider,
			source_image,
			target_image,
			target_video,
			face_recognition_dropdown,
			reference_face_position_gallery_index,
			face_analyser_direction,
			face_analyser_age,
			face_analyser_gender,
			frame_processors_checkbox_group,
			face_swapper_model_dropdown,
			face_enhancer_model_dropdown,
			frame_enhancer_model_dropdown,
			reference_face_distance_slider,
			face_enhancer_blend_slider,
			frame_enhancer_blend_slider,
			output_image_quality_slider,
			temp_frame_format_dropdown,
			temp_frame_quality_slider,
			output_video_encoder_dropdown,
			output_video_quality_slider,
			trim_frame_start_slider,
			trim_frame_end_slider,
			common_options_checkbox_group,
		)
		if is_image(output_path):
			return gradio.update(value = output_path, visible = True), gradio.update(value = None, visible = False)
		if is_video(output_path):
			return gradio.update(value = None, visible = False), gradio.update(value = output_path, visible = True)
		return gradio.update(), gradio.update()


def clear() -> Tuple[Update, Update]:
	if facefusion.globals.target_path:
		clear_temp(facefusion.globals.target_path)
	return gradio.update(value = None), gradio.update(value = None)
