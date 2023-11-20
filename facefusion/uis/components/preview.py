from typing import Any, Dict, List, Optional
from uuid import uuid4

import cv2
import gradio
from starlette.datastructures import MutableHeaders

import facefusion.globals
from facefusion import wording
from facefusion.typing import FaceRecognition, Frame, Face, FaceAnalyserAge, FaceAnalyserDirection, FaceAnalyserGender
from facefusion.vision import get_video_frame, count_video_frame_total, normalize_frame_color, resize_frame_dimension, read_static_image
from facefusion.face_analyser import get_one_face
from facefusion.face_reference import get_face_reference, set_face_reference
from facefusion.predictor import predict_frame
from facefusion.processors.frame.core import load_frame_processor_module
from facefusion.utilities import is_video, is_image
from facefusion.uis.typing import ComponentName, Update
from facefusion.uis.core import get_ui_component, register_ui_component
from modules.system_monitor import monitor_call_context

PREVIEW_IMAGE : Optional[gradio.Image] = None
PREVIEW_FRAME_SLIDER : Optional[gradio.Slider] = None

def _get_consume_text(consume: int) -> str:
	text = f"{consume} credit" if consume == 1 else f"{consume} credits"
	return f"{wording.get('preview_image_label')} -- Each preview image costs {text}"

def render() -> None:
	global PREVIEW_IMAGE
	global PREVIEW_FRAME_SLIDER

	preview_image_args: Dict[str, Any] =\
	{
		'label': _get_consume_text(1),
		'interactive': False
	}
	preview_frame_slider_args: Dict[str, Any] =\
	{
		'label': wording.get('preview_frame_slider_label'),
		'step': 1,
		'minimum': 0,
		'maximum': 100,
		'visible': False
	}
	# conditional_set_face_reference()
	# source_face = get_one_face(read_static_image(facefusion.globals.source_path))
	# reference_face = get_face_reference() if 'reference' in facefusion.globals.face_recognition else None
	# if is_image(facefusion.globals.target_path):
	# 	target_frame = read_static_image(facefusion.globals.target_path)
	# 	preview_frame = process_preview_frame(source_face, reference_face, target_frame)
	# 	preview_image_args['value'] = normalize_frame_color(preview_frame)
	# if is_video(facefusion.globals.target_path):
	# 	temp_frame = get_video_frame(facefusion.globals.target_path, facefusion.globals.reference_frame_number)
	# 	preview_frame = process_preview_frame(source_face, reference_face, temp_frame)
	# 	preview_image_args['value'] = normalize_frame_color(preview_frame)
	# 	preview_image_args['visible'] = True
	# 	preview_frame_slider_args['value'] = facefusion.globals.reference_frame_number
	# 	preview_frame_slider_args['maximum'] = count_video_frame_total(facefusion.globals.target_path)
	# 	preview_frame_slider_args['visible'] = True
	PREVIEW_IMAGE = gradio.Image(**preview_image_args)
	PREVIEW_FRAME_SLIDER = gradio.Slider(**preview_frame_slider_args)
	register_ui_component('preview_frame_slider', PREVIEW_FRAME_SLIDER)


def listen() -> None:
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

	frame_processors_checkbox_group.change(
		lambda processors: gradio.update(label=_get_consume_text(len(processors))),
		inputs=frame_processors_checkbox_group,
		outputs=PREVIEW_IMAGE,
	)

	PREVIEW_FRAME_SLIDER.change(
		update_preview_image_wrapper,
		inputs = [
			source_image,
			target_image,
			target_video,
			PREVIEW_FRAME_SLIDER,
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
		],
		outputs = PREVIEW_IMAGE
	)
	multi_component_names : List[ComponentName] =\
	[
		'source_image',
		'target_image',
		'target_video'
	]
	for component_name in multi_component_names:
		component = get_ui_component(component_name)
		if component:
			for method in [ 'upload', 'change', 'clear' ]:
				getattr(component, method)(
					update_preview_image_wrapper,
					inputs = [
						source_image,
						target_image,
						target_video,
						PREVIEW_FRAME_SLIDER,
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
					],
					outputs = PREVIEW_IMAGE)
				if component_name == "source_image":
					continue
				getattr(component, method)(
					update_preview_frame_slider,
					inputs = [target_image, target_video],
					outputs = PREVIEW_FRAME_SLIDER)
	update_component_names : List[ComponentName] =\
	[
		'face_recognition_dropdown',
		'frame_processors_checkbox_group',
		'face_swapper_model_dropdown',
		'face_enhancer_model_dropdown',
		'frame_enhancer_model_dropdown',
		'reference_face_position_gallery_index',
	]
	for component_name in update_component_names:
		component = get_ui_component(component_name)
		if component:
			component.change(
				update_preview_image_wrapper,
				inputs = [
					source_image,
					target_image,
					target_video,
					PREVIEW_FRAME_SLIDER,
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
				],
				outputs = PREVIEW_IMAGE)
	select_component_names : List[ComponentName] =\
	[
		# 'reference_face_position_gallery',
		'face_analyser_direction_dropdown',
		'face_analyser_age_dropdown',
		'face_analyser_gender_dropdown'
	]
	for component_name in select_component_names:
		component = get_ui_component(component_name)
		if component:
			component.select(
				update_preview_image_wrapper,
				inputs = [
					source_image,
					target_image,
					target_video,
					PREVIEW_FRAME_SLIDER,
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
				],
				outputs = PREVIEW_IMAGE
			)
	change_component_names : List[ComponentName] =\
	[
		'reference_face_distance_slider',
		'face_enhancer_blend_slider',
		'frame_enhancer_blend_slider'
	]
	for component_name in change_component_names:
		component = get_ui_component(component_name)
		if component:
			component.change(
				update_preview_image_wrapper,
				inputs = [
					source_image,
					target_image,
					target_video,
					PREVIEW_FRAME_SLIDER,
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
				],
				outputs = PREVIEW_IMAGE
			)

def update_preview_image_wrapper(
	request: gradio.Request,
	source_image: Frame | None,
	target_image: Frame | None,
	target_video: str | None,
	reference_frame_number: int,
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
) -> Update:
	if source_image is None or (target_image is None and target_video is None):
		return

	source_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
	if target_image is not None:
		target_image = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)

	task_id = str(uuid4())
	MutableHeaders.__setitem__(request.headers, "x-task-id", task_id)

	with monitor_call_context(
		request,
		"extensions.facefusion",
		"extensions.facefusion",
		task_id,
		decoded_params={
			"width": 640,
			"height": 640,
			"n_iter": 1,
		},
		is_intermediate=False,
	):
		return update_preview_image(
			request,
			source_image,
			target_image,
			target_video,
			reference_frame_number,
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
		)

def update_preview_image(
	request: gradio.Request,
	source_image: Frame,
	target_image: Frame | None,
	target_video: str | None,
	reference_frame_number: int,
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
) -> Update:
	# conditional_set_face_reference()
	# source_face = get_one_face(read_static_image(facefusion.globals.source_path))
	# reference_face = get_face_reference() if 'reference' in facefusion.globals.face_recognition else None
	source_face = get_one_face(
		source_image,
		0,
		face_analyser_direction,
		face_analyser_age,
		face_analyser_gender,
	)
	reference_face = conditional_get_face_reference(
		face_recognition_dropdown,
		target_image,
		target_video,
		reference_frame_number,
		reference_face_position_gallery_index,
		face_analyser_direction,
		face_analyser_age,
		face_analyser_gender,
	)

	if target_image is not None:
		preview_frame = process_preview_frame(
			request,
			source_face,
			reference_face,
			target_image,
			frame_processors_checkbox_group,
			face_swapper_model_dropdown,
			face_enhancer_model_dropdown,
			frame_enhancer_model_dropdown,
			source_image,
			target_image,
			target_video,
			face_recognition_dropdown,
			reference_face_distance_slider,
			face_analyser_direction,
			face_analyser_age,
			face_analyser_gender,
			face_enhancer_blend_slider,
			frame_enhancer_blend_slider,
		)
		preview_frame = normalize_frame_color(preview_frame)
		return gradio.update(value = preview_frame)

	if target_video is not None:
		temp_frame = get_video_frame(target_video, reference_frame_number)
		preview_frame = process_preview_frame(
			request,
			source_face,
			reference_face,
			temp_frame,
			frame_processors_checkbox_group,
			face_swapper_model_dropdown,
			face_enhancer_model_dropdown,
			frame_enhancer_model_dropdown,
			source_image,
			target_image,
			target_video,
			face_recognition_dropdown,
			reference_face_distance_slider,
			face_analyser_direction,
			face_analyser_age,
			face_analyser_gender,
			face_enhancer_blend_slider,
			frame_enhancer_blend_slider,
		)
		preview_frame = normalize_frame_color(preview_frame)
		return gradio.update(value = preview_frame)
	return gradio.update(value = None)


def update_preview_frame_slider(target_image: Frame | None, target_video: str | None) -> Update:
	if target_image is not None:
		return gradio.update(value = None, maximum = None, visible = False)
	if target_video is not None:
		video_frame_total = count_video_frame_total(target_video)
		return gradio.update(maximum = video_frame_total, visible = True)
	return gradio.update(value = None, maximum = None, visible = False)



def process_preview_frame(
	request: gradio.Request,
	source_face: Face,
	reference_face: Face,
	temp_frame: Frame,
	frame_processors_checkbox_group: list[str],
	face_swapper_model_dropdown: str,
	face_enhancer_model_dropdown: str,
	frame_enhancer_model_dropdown: str,
	source_image: Frame | None,
	target_image: Frame | None,
	target_video: str | None,
	face_recognition_dropdown: FaceRecognition,
	reference_face_distance_slider: float,
	face_analyser_direction: FaceAnalyserDirection,
	face_analyser_age: FaceAnalyserAge,
	face_analyser_gender: FaceAnalyserGender,
	face_enhancer_blend_slider: int,
	frame_enhancer_blend_slider: int,

) -> Frame:
	temp_frame = resize_frame_dimension(temp_frame, 640, 640)
	# if predict_frame(temp_frame):
	# 	return cv2.GaussianBlur(temp_frame, (99, 99), 0)

	kwargs = {
		"face_swapper_model": face_swapper_model_dropdown,
		"face_enhancer_model": face_enhancer_model_dropdown,
		"frame_enhancer_model": frame_enhancer_model_dropdown,
		"source_image": source_image,
		"target_image": target_image,
		"target_video": target_video,
		"face_recognition_dropdown": face_recognition_dropdown,
		"reference_face_distance_slider": reference_face_distance_slider,
		"face_analyser_direction": face_analyser_direction,
		"face_analyser_age": face_analyser_age,
		"face_analyser_gender": face_analyser_gender,
		"face_enhancer_blend_slider": face_enhancer_blend_slider,
		"frame_enhancer_blend_slider": frame_enhancer_blend_slider,
	}
	with monitor_call_context(
		request,
		"extensions.facefusion",
		"extensions.facefusion",
		decoded_params={
			"width": 640,
			"height": 640,
			"n_iter": len(frame_processors_checkbox_group),
		},
	):
		for frame_processor in frame_processors_checkbox_group:
			frame_processor_module = load_frame_processor_module(frame_processor)
			if frame_processor_module.pre_process('preview', kwargs):
				frame_processor_module.get_frame_processor(kwargs)
				temp_frame = frame_processor_module.process_frame(
					source_face,
					reference_face,
					temp_frame,
					kwargs,
				)
	return temp_frame


def conditional_set_face_reference() -> None:
	if 'reference' in facefusion.globals.face_recognition and not get_face_reference():
		reference_frame = get_video_frame(facefusion.globals.target_path, facefusion.globals.reference_frame_number)
		reference_face = get_one_face(reference_frame, facefusion.globals.reference_face_position)
		set_face_reference(reference_face)

def conditional_get_face_reference(
	face_recognition_dropdown: FaceRecognition,
	target_image: Frame | None,
	target_video: str | None,
	reference_frame_number: int,
	reference_face_position_gallery_index: int,
	face_analyser_direction: FaceAnalyserDirection,
	face_analyser_age: FaceAnalyserAge,
	face_analyser_gender: FaceAnalyserGender,
) -> Face | None:
	if face_recognition_dropdown == "many":
		return None

	if target_image is not None:
		reference_frame = target_image
	elif target_video is not None:
		reference_frame = get_video_frame(target_video, reference_frame_number)

	return get_one_face(
		reference_frame,
		reference_face_position_gallery_index,
		face_analyser_direction,
		face_analyser_age,
		face_analyser_gender,
	)
