from typing import Any, List, Dict, Literal, Optional
from argparse import ArgumentParser
import insightface
import threading

import facefusion.globals
import facefusion.processors.frame.core as frame_processors
from facefusion import wording
from facefusion.core import update_status
from facefusion.face_analyser import get_one_face, get_many_faces, find_similar_faces, clear_face_analyser
from facefusion.face_reference import get_face_reference, set_face_reference
from facefusion.typing import Face, FaceAnalyserAge, FaceAnalyserDirection, FaceAnalyserGender, FaceRecognition, Frame, Update_Process, ProcessMode, ModelValue, OptionsWithModel
from facefusion.utilities import conditional_download, resolve_relative_path, is_image, is_video, is_file, is_download_done
from facefusion.vision import read_image, read_static_image, write_image, get_video_frame
from facefusion.processors.frame import globals as frame_processors_globals
from facefusion.processors.frame import choices as frame_processors_choices

FRAME_PROCESSOR = None
THREAD_LOCK : threading.Lock = threading.Lock()
NAME = 'FACEFUSION.FRAME_PROCESSOR.FACE_SWAPPER'
MODELS : Dict[str, ModelValue] =\
{
	'inswapper_128':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx',
		'path': resolve_relative_path('../.assets/models/inswapper_128.onnx')
	},
	'inswapper_128_fp16':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx',
		'path': resolve_relative_path('../.assets/models/inswapper_128_fp16.onnx')
	}
}
OPTIONS : Optional[OptionsWithModel] = None


# def get_frame_processor() -> Any:
# 	global FRAME_PROCESSOR

# 	with THREAD_LOCK:
# 		if FRAME_PROCESSOR is None:
# 			model_path = get_options('model').get('path')
# 			FRAME_PROCESSOR = insightface.model_zoo.get_model(model_path, providers = facefusion.globals.execution_providers)
# 	return FRAME_PROCESSOR

def get_frame_processor(kwargs: dict[str, Any]) -> None:
	model_name = kwargs["face_swapper_model"]
	model_path = MODELS[model_name]["path"]
	kwargs["face_swapper_frame_processor"] = insightface.model_zoo.get_model(model_path, providers = facefusion.globals.execution_providers)


def clear_frame_processor() -> None:
	global FRAME_PROCESSOR

	FRAME_PROCESSOR = None


def get_options(key : Literal[ 'model' ]) -> Any:
	global OPTIONS

	if OPTIONS is None:
		OPTIONS = \
		{
			'model': MODELS[frame_processors_globals.face_swapper_model]
		}
	return OPTIONS.get(key)


def set_options(key : Literal[ 'model' ], value : Any) -> None:
	global OPTIONS

	OPTIONS[key] = value


def register_args(program : ArgumentParser) -> None:
	program.add_argument('--face-swapper-model', help = wording.get('frame_processor_model_help'), dest = 'face_swapper_model', default = 'inswapper_128', choices = frame_processors_choices.face_swapper_models)


def apply_args(program : ArgumentParser) -> None:
	args = program.parse_args([])
	frame_processors_globals.face_swapper_model = args.face_swapper_model


def pre_check() -> bool:
	if not facefusion.globals.skip_download:
		download_directory_path = resolve_relative_path('../.assets/models')
		model_url = get_options('model').get('url')
		conditional_download(download_directory_path, [ model_url ])
	return True


def pre_process(mode : ProcessMode, kwargs: dict[str, Any]) -> bool:
	model_name = kwargs["face_swapper_model"]
	model_url = MODELS[model_name]["url"]
	model_path = MODELS[model_name]["path"]
	if not facefusion.globals.skip_download and not is_download_done(model_url, model_path):
		update_status(wording.get('model_download_not_done') + wording.get('exclamation_mark'), NAME)
		return False
	elif not is_file(model_path):
		update_status(wording.get('model_file_not_present') + wording.get('exclamation_mark'), NAME)
		return False
	if kwargs["source_image"] is None:
		update_status(wording.get('select_image_source') + wording.get('exclamation_mark'), NAME)
		return False
	elif not get_one_face(
		kwargs["source_image"],
		0,
		kwargs["face_analyser_direction"],
		kwargs["face_analyser_age"],
		kwargs["face_analyser_gender"],
	):
		update_status(wording.get('no_source_face_detected') + wording.get('exclamation_mark'), NAME)
		return False
	if mode in [ 'output', 'preview' ] and kwargs["target_image"] is None and kwargs["target_video"] is None:
		update_status(wording.get('select_image_or_video_target') + wording.get('exclamation_mark'), NAME)
		return False
	if mode == 'output' and not facefusion.globals.output_path:
		update_status(wording.get('select_file_or_directory_output') + wording.get('exclamation_mark'), NAME)
		return False
	return True


def post_process() -> None:
	clear_frame_processor()
	clear_face_analyser()
	read_static_image.cache_clear()


def swap_face(source_face : Face, target_face : Face, temp_frame : Frame, frame_processor: Any) -> Frame:
	return frame_processor.get(temp_frame, target_face, source_face, paste_back = True)


def process_frame(source_face : Face, reference_face : Face, temp_frame : Frame, kwargs: dict[str, Any]) -> Frame:
	if kwargs["face_recognition_dropdown"] == "reference":
		similar_faces = find_similar_faces(
			temp_frame,
			reference_face,
			kwargs["reference_face_distance_slider"],
			kwargs["face_analyser_direction"],
			kwargs["face_analyser_age"],
			kwargs["face_analyser_gender"],
		)
		if similar_faces:
			for similar_face in similar_faces:
				temp_frame = swap_face(source_face, similar_face, temp_frame, kwargs["face_swapper_frame_processor"])
	else:
		many_faces = get_many_faces(
			temp_frame,
			kwargs["face_analyser_direction"],
			kwargs["face_analyser_age"],
			kwargs["face_analyser_gender"],
		)
		if many_faces:
			for target_face in many_faces:
				temp_frame = swap_face(source_face, target_face, temp_frame, kwargs["face_swapper_frame_processor"])
	return temp_frame


def process_frames(
	reference_face: Face | None,
	temp_frame_paths: List[str],
	update_progress : Update_Process,
	kwargs: dict[str, Any],
) -> None:
	source_face = get_one_face(
		kwargs["source_image"],
		0,
		kwargs["face_analyser_direction"],
		kwargs["face_analyser_age"],
		kwargs["face_analyser_gender"],
	)
	for temp_frame_path in temp_frame_paths:
		temp_frame = read_image(temp_frame_path)
		result_frame = process_frame(source_face, reference_face, temp_frame, kwargs)
		write_image(temp_frame_path, result_frame)
		update_progress()


def process_image(source_path : str, target_path : str, output_path : str) -> None:
	source_face = get_one_face(read_static_image(source_path))
	target_frame = read_static_image(target_path)
	reference_face = get_one_face(target_frame, facefusion.globals.reference_face_position) if 'reference' in facefusion.globals.face_recognition else None
	result_frame = process_frame(source_face, reference_face, target_frame)
	write_image(output_path, result_frame)

def process_image_frame(frame: Frame, kwargs: dict[str, Any]) -> Frame:
	source_face = get_one_face(
		kwargs["source_image"],
		0,
		kwargs["face_analyser_direction"],
		kwargs["face_analyser_age"],
		kwargs["face_analyser_gender"],
	)
	if kwargs["face_recognition_dropdown"] == "reference":
		reference_face = get_one_face(
			frame,
			kwargs["reference_face_position_gallery_index"],
			kwargs["face_analyser_direction"],
			kwargs["face_analyser_age"],
			kwargs["face_analyser_gender"],
		)
	else:
		reference_face = None

	return process_frame(source_face, reference_face, frame, kwargs)


def process_video(source_path : str, temp_frame_paths : List[str]) -> None:
	conditional_set_face_reference(temp_frame_paths)
	frame_processors.multi_process_frames(source_path, temp_frame_paths, process_frames)

def process_video_frame(temp_frame_paths: List[str], kwargs: dict[str, Any]) -> None:
	if kwargs["face_recognition_dropdown"] == "reference":
		reference_frame = get_video_frame(kwargs["target_video"], kwargs["preview_frame_slider"])
		reference_face = get_one_face(
			reference_frame,
			kwargs["reference_face_position_gallery_index"],
			kwargs["face_analyser_direction"],
			kwargs["face_analyser_age"],
			kwargs["face_analyser_gender"],
		)
	else:
		reference_face = None

	frame_processors.multi_process_frames(reference_face, temp_frame_paths, process_frames, kwargs)


def conditional_set_face_reference(temp_frame_paths : List[str]) -> None:
	if 'reference' in facefusion.globals.face_recognition and not get_face_reference():
		reference_frame = read_static_image(temp_frame_paths[facefusion.globals.reference_frame_number])
		reference_face = get_one_face(reference_frame, facefusion.globals.reference_face_position)
		set_face_reference(reference_face)
