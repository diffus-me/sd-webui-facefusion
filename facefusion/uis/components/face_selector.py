from typing import List, Optional, Tuple, Any, Dict

import gradio
import cv2

import facefusion.choices
import facefusion.globals
from facefusion import wording
from facefusion.vision import get_video_frame, normalize_frame_color, read_static_image
from facefusion.face_analyser import get_many_faces
from facefusion.face_reference import clear_face_reference
from facefusion.typing import FaceAnalyserAge, FaceAnalyserGender, Frame, FaceRecognition, FaceAnalyserDirection
from facefusion.utilities import is_image, is_video
from facefusion.uis.core import get_ui_component, register_ui_component
from facefusion.uis.typing import ComponentName, Update

FACE_RECOGNITION_DROPDOWN : Optional[gradio.Dropdown] = None
REFERENCE_FACE_POSITION_GALLERY : Optional[gradio.Gallery] = None
REFERENCE_FACE_DISTANCE_SLIDER : Optional[gradio.Slider] = None
REFERENCE_FACE_POSITION_GALLERY_INDEX : Optional[gradio.Number] = None


def render() -> None:
	global FACE_RECOGNITION_DROPDOWN
	global REFERENCE_FACE_POSITION_GALLERY
	global REFERENCE_FACE_DISTANCE_SLIDER
	global REFERENCE_FACE_POSITION_GALLERY_INDEX

	reference_face_gallery_args: Dict[str, Any] =\
	{
		'label': wording.get('reference_face_gallery_label'),
		'height': 120,
		'object_fit': 'cover',
		'columns': 10,
		'allow_preview': False,
		'visible': 'reference' in facefusion.globals.face_recognition
	}
	# if is_image(facefusion.globals.target_path):
	# 	reference_frame = read_static_image(facefusion.globals.target_path)
	# 	reference_face_gallery_args['value'] = extract_gallery_frames(reference_frame)
	# if is_video(facefusion.globals.target_path):
	# 	reference_frame = get_video_frame(facefusion.globals.target_path, facefusion.globals.reference_frame_number)
	# 	reference_face_gallery_args['value'] = extract_gallery_frames(reference_frame)
	FACE_RECOGNITION_DROPDOWN = gradio.Dropdown(
		label = wording.get('face_recognition_dropdown_label'),
		choices = facefusion.choices.face_recognitions,
		value = facefusion.globals.face_recognition
	)
	REFERENCE_FACE_POSITION_GALLERY = gradio.Gallery(**reference_face_gallery_args)
	REFERENCE_FACE_DISTANCE_SLIDER = gradio.Slider(
		label = wording.get('reference_face_distance_slider_label'),
		value = facefusion.globals.reference_face_distance,
		step = 0.05,
		minimum = 0,
		maximum = 3,
		visible = 'reference' in facefusion.globals.face_recognition
	)
	REFERENCE_FACE_POSITION_GALLERY_INDEX = gradio.Number(value=0, precision=0, visible=False)

	register_ui_component('face_recognition_dropdown', FACE_RECOGNITION_DROPDOWN)
	register_ui_component('reference_face_position_gallery', REFERENCE_FACE_POSITION_GALLERY)
	register_ui_component('reference_face_distance_slider', REFERENCE_FACE_DISTANCE_SLIDER)
	register_ui_component('reference_face_position_gallery_index', REFERENCE_FACE_POSITION_GALLERY_INDEX)


def listen() -> None:
	FACE_RECOGNITION_DROPDOWN.select(update_face_recognition, inputs = FACE_RECOGNITION_DROPDOWN, outputs = [ REFERENCE_FACE_POSITION_GALLERY, REFERENCE_FACE_DISTANCE_SLIDER ])
	REFERENCE_FACE_POSITION_GALLERY.select(update_gallery_index, outputs=[REFERENCE_FACE_POSITION_GALLERY_INDEX])
	# REFERENCE_FACE_POSITION_GALLERY.select(clear_and_update_face_reference_position)
	# REFERENCE_FACE_DISTANCE_SLIDER.change(update_reference_face_distance, inputs = REFERENCE_FACE_DISTANCE_SLIDER)
	target_image = get_ui_component("target_image")
	target_video = get_ui_component("target_video")
	preview_frame_slider = get_ui_component("preview_frame_slider")
	face_analyser_direction = get_ui_component("face_analyser_direction_dropdown")
	face_analyser_age = get_ui_component("face_analyser_age_dropdown")
	face_analyser_gender = get_ui_component("face_analyser_gender_dropdown")

	multi_component_names : List[ComponentName] =\
	[
		# 'source_image',
		'target_image',
		'target_video'
	]
	for component_name in multi_component_names:
		component = get_ui_component(component_name)
		if component:
			for method in [ 'upload', 'change', 'clear' ]:
				getattr(component, method)(
					update_face_reference_position,
					inputs = [
						target_image,
						target_video,
						preview_frame_slider,
						face_analyser_direction,
						face_analyser_age,
						face_analyser_gender,
					],
					outputs = REFERENCE_FACE_POSITION_GALLERY,
				)
	select_component_names : List[ComponentName] =\
	[
		'face_analyser_direction_dropdown',
		'face_analyser_age_dropdown',
		'face_analyser_gender_dropdown'
	]
	for component_name in select_component_names:
		component = get_ui_component(component_name)
		if component:
			component.select(
				update_face_reference_position,
				inputs = [
					target_image,
					target_video,
					preview_frame_slider,
					face_analyser_direction,
					face_analyser_age,
					face_analyser_gender,
				],
				outputs = REFERENCE_FACE_POSITION_GALLERY,
			)
	if preview_frame_slider:
		preview_frame_slider.release(
			update_face_reference_position,
			inputs = [
				target_image,
				target_video,
				preview_frame_slider,
				face_analyser_direction,
				face_analyser_age,
				face_analyser_gender,
			],
			outputs = REFERENCE_FACE_POSITION_GALLERY,
		)


def update_face_recognition(face_recognition : FaceRecognition) -> Tuple[Update, Update]:
	if face_recognition == 'reference':
		# facefusion.globals.face_recognition = face_recognition
		return gradio.update(visible = True), gradio.update(visible = True)
	if face_recognition == 'many':
		# facefusion.globals.face_recognition = face_recognition
		return gradio.update(visible = False), gradio.update(visible = False)


def clear_and_update_face_reference_position(event: gradio.SelectData) -> Update:
	clear_face_reference()
	return update_face_reference_position(event.index)

def update_gallery_index(event: gradio.SelectData) -> Update:
	return event.index


def update_face_reference_position(
	target_image: Frame | None,
	target_video: str,
	preview_frame_slider: int,
	face_analyser_direction: FaceAnalyserDirection,
	face_analyser_age: FaceAnalyserAge,
	face_analyser_gender: FaceAnalyserGender,
) -> Update:
	gallery_frames = []
	if target_image is not None:
		reference_frame = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
		gallery_frames = extract_gallery_frames(
			reference_frame,
			face_analyser_direction,
			face_analyser_age,
			face_analyser_gender,
		)
	elif target_video is not None:
		reference_frame = get_video_frame(target_video, preview_frame_slider)
		gallery_frames = extract_gallery_frames(
			reference_frame,
			face_analyser_direction,
			face_analyser_age,
			face_analyser_gender,
		)
	if gallery_frames:
		return gradio.update(value = gallery_frames)
	return gradio.update(value = None)


def update_reference_face_distance(reference_face_distance : float) -> Update:
	facefusion.globals.reference_face_distance = reference_face_distance
	return gradio.update(value = reference_face_distance)


def extract_gallery_frames(
	reference_frame: Frame,
	face_analyser_direction: FaceAnalyserDirection,
	face_analyser_age: FaceAnalyserAge,
	face_analyser_gender: FaceAnalyserGender,
) -> List[Frame]:
	crop_frames = []
	faces = get_many_faces(
		reference_frame,
		face_analyser_direction,
		face_analyser_age,
		face_analyser_gender,
	)
	for face in faces:
		start_x, start_y, end_x, end_y = map(int, face['bbox'])
		padding_x = int((end_x - start_x) * 0.25)
		padding_y = int((end_y - start_y) * 0.25)
		start_x = max(0, start_x - padding_x)
		start_y = max(0, start_y - padding_y)
		end_x = max(0, end_x + padding_x)
		end_y = max(0, end_y + padding_y)
		crop_frame = reference_frame[start_y:end_y, start_x:end_x]
		crop_frame = normalize_frame_color(crop_frame)
		crop_frames.append(crop_frame)
	return crop_frames
