from typing import Any, List, Dict, Literal, Optional
from argparse import ArgumentParser
import threading
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

import facefusion.globals
import facefusion.processors.frame.core as frame_processors
from facefusion import wording
from facefusion.core import update_status
from facefusion.face_analyser import clear_face_analyser
from facefusion.typing import Frame, Face, Update_Process, ProcessMode, ModelValue, OptionsWithModel
from facefusion.utilities import conditional_download, resolve_relative_path, is_file, is_download_done, get_device, update_model_path
from facefusion.vision import read_image, read_static_image, write_image
from facefusion.processors.frame import globals as frame_processors_globals
from facefusion.processors.frame import choices as frame_processors_choices

FRAME_PROCESSOR = None
THREAD_SEMAPHORE : threading.Semaphore = threading.Semaphore()
THREAD_LOCK : threading.Lock = threading.Lock()
NAME = 'FACEFUSION.FRAME_PROCESSOR.FRAME_ENHANCER'
MODELS: Dict[str, ModelValue] =\
{
	'realesrgan_x2plus':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/RealESRGAN_x2plus.pth',
		'path': resolve_relative_path('../.assets/models/RealESRGAN_x2plus.pth'),
		'scale': 2
	},
	'realesrgan_x4plus':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/RealESRGAN_x4plus.pth',
		'path': resolve_relative_path('../.assets/models/RealESRGAN_x4plus.pth'),
		'scale': 4
	},
	'realesrnet_x4plus':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/RealESRNet_x4plus.pth',
		'path': resolve_relative_path('../.assets/models/RealESRNet_x4plus.pth'),
		'scale': 4
	}
}
OPTIONS : Optional[OptionsWithModel] = None

update_model_path(MODELS)

# def get_frame_processor() -> Any:
# 	global FRAME_PROCESSOR

# 	with THREAD_LOCK:
# 		if FRAME_PROCESSOR is None:
# 			model_path = get_options('model').get('path')
# 			model_scale = get_options('model').get('scale')
# 			FRAME_PROCESSOR = RealESRGANer(
# 				model_path = model_path,
# 				model = RRDBNet(
# 					num_in_ch = 3,
# 					num_out_ch = 3,
# 					scale = model_scale
# 				),
# 				device = get_device(facefusion.globals.execution_providers),
# 				scale = model_scale
# 			)
# 	return FRAME_PROCESSOR

def get_frame_processor(kwargs: dict[str, Any]) -> None:
	model_name = kwargs["frame_enhancer_model"]
	model_path = MODELS[model_name]["path"]
	model_scale = MODELS[model_name]["scale"]
	kwargs["frame_enhancer_frame_processor"] = RealESRGANer(
			model_path = model_path,
			model = RRDBNet(
				num_in_ch = 3,
				num_out_ch = 3,
				scale = model_scale
			),
			device = get_device(facefusion.globals.execution_providers),
			scale = model_scale
		)


def clear_frame_processor() -> None:
	global FRAME_PROCESSOR

	FRAME_PROCESSOR = None


def get_options(key : Literal[ 'model' ]) -> Any:
	global OPTIONS

	if OPTIONS is None:
		OPTIONS = \
		{
			'model': MODELS[frame_processors_globals.frame_enhancer_model]
		}
	return OPTIONS.get(key)


def set_options(key : Literal[ 'model' ], value : Any) -> None:
	global OPTIONS

	OPTIONS[key] = value


def register_args(program : ArgumentParser) -> None:
	program.add_argument('--frame-enhancer-model', help = wording.get('frame_processor_model_help'), dest = 'frame_enhancer_model', default = 'realesrgan_x2plus', choices = frame_processors_choices.frame_enhancer_models)
	program.add_argument('--frame-enhancer-blend', help = wording.get('frame_processor_blend_help'), dest = 'frame_enhancer_blend', type = int, default = 100, choices = range(101), metavar = '[0-100]')


def apply_args(program : ArgumentParser) -> None:
	args = program.parse_args([])
	frame_processors_globals.frame_enhancer_model = args.frame_enhancer_model
	frame_processors_globals.frame_enhancer_blend = args.frame_enhancer_blend


def pre_check() -> bool:
	if not facefusion.globals.skip_download:
		download_directory_path = resolve_relative_path('../.assets/models')
		model_url = get_options('model').get('url')
		conditional_download(download_directory_path, [ model_url ])
	return True


def pre_process(mode : ProcessMode, kwargs: dict[str, Any]) -> bool:
	model_name = kwargs["frame_enhancer_model"]
	model_url = MODELS[model_name]["url"]
	model_path = MODELS[model_name]["path"]
	if not facefusion.globals.skip_download and not is_download_done(model_url, model_path):
		update_status(wording.get('model_download_not_done') + wording.get('exclamation_mark'), NAME)
		return False
	elif not is_file(model_path):
		update_status(wording.get('model_file_not_present') + wording.get('exclamation_mark'), NAME)
		return False
	if mode == 'output' and not facefusion.globals.output_path:
		update_status(wording.get('select_file_or_directory_output') + wording.get('exclamation_mark'), NAME)
		return False
	return True


def post_process() -> None:
	clear_frame_processor()
	clear_face_analyser()
	read_static_image.cache_clear()


def enhance_frame(temp_frame : Frame, frame_processor: Any, blend: int) -> Frame:
	with THREAD_SEMAPHORE:
		paste_frame, _ = frame_processor.enhance(temp_frame)
		temp_frame = blend_frame(temp_frame, paste_frame, blend)
	return temp_frame


def blend_frame(temp_frame : Frame, paste_frame : Frame, blend: int) -> Frame:
	frame_enhancer_blend = 1 - (blend / 100)
	temp_frame = cv2.resize(temp_frame, (paste_frame.shape[1], paste_frame.shape[0]))
	temp_frame = cv2.addWeighted(temp_frame, frame_enhancer_blend, paste_frame, 1 - frame_enhancer_blend, 0)
	return temp_frame


def process_frame(source_face : Face, reference_face : Face, temp_frame : Frame, kwargs: dict[str, Any]) -> Frame:
	return enhance_frame(
		temp_frame,
		kwargs["frame_enhancer_frame_processor"],
		kwargs["frame_enhancer_blend_slider"],
	)


def process_frames(
	reference_face: Face | None,
	temp_frame_paths: List[str],
	update_progress : Update_Process,
	kwargs: dict[str, Any],
) -> None:
	for temp_frame_path in temp_frame_paths:
		temp_frame = read_image(temp_frame_path)
		result_frame = process_frame(None, None, temp_frame, kwargs)
		write_image(temp_frame_path, result_frame)
		update_progress()


def process_image(source_path : str, target_path : str, output_path : str) -> None:
	target_frame = read_static_image(target_path)
	result = process_frame(None, None, target_frame)
	write_image(output_path, result)

def process_image_frame(frame: Frame, kwargs: dict[str, Any]) -> None:
	return process_frame(None, None, frame, kwargs)


def process_video(source_path : str, temp_frame_paths : List[str]) -> None:
	frame_processors.multi_process_frames(None, temp_frame_paths, process_frames)


def process_video_frame(temp_frame_paths: List[str], kwargs: dict[str, Any]) -> None:
	frame_processors.multi_process_frames(None, temp_frame_paths, process_frames, kwargs)
