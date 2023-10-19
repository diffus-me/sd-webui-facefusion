import os
from typing import Any
from pathlib import Path

from facefusion.vision import write_image

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import signal
import sys
import warnings
import platform
import shutil
import onnxruntime
import tensorflow
from argparse import ArgumentParser, HelpFormatter
import gradio

import facefusion.choices
import facefusion.globals
from facefusion import metadata, wording
from facefusion.predictor import predict_frame, predict_image, predict_video
from facefusion.processors.frame.core import get_frame_processors_modules, load_frame_processor_module
from facefusion.utilities import is_image, is_video, detect_fps, compress_image, merge_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clear_temp, list_module_names, encode_execution_providers, decode_execution_providers, normalize_output_path
from facefusion.utilities import TEMP_DIRECTORY_PATH, create_temp_dir, get_temp_video_frame_paths, rename_temp_video_to_output, clear_temp_dir
from facefusion.typing import FaceRecognition, Frame, FaceAnalyserAge, FaceAnalyserDirection, FaceAnalyserGender
from modules.system_monitor import monitor_call_context

warnings.filterwarnings('ignore', category = FutureWarning, module = 'insightface')
warnings.filterwarnings('ignore', category = UserWarning, module = 'gradio')
warnings.filterwarnings('ignore', category = UserWarning, module = 'torchvision')

def get_argument_parser():
	signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
	program = ArgumentParser(formatter_class = lambda prog: HelpFormatter(prog, max_help_position = 120), add_help = False)
	# general
	program.add_argument('-s', '--source', help = wording.get('source_help'), dest = 'source_path')
	program.add_argument('-t', '--target', help = wording.get('target_help'), dest = 'target_path')
	program.add_argument('-o', '--output', help = wording.get('output_help'), dest = 'output_path')
	program.add_argument('-v', '--version', version = metadata.get('name') + ' ' + metadata.get('version'), action = 'version')
	# misc
	group_misc = program.add_argument_group('misc')
	group_misc.add_argument('--skip-download', help = wording.get('skip_download_help'), dest = 'skip_download', action = 'store_true')
	group_misc.add_argument('--headless', help = wording.get('headless_help'), dest = 'headless', action = 'store_true')
	# execution
	group_execution = program.add_argument_group('execution')
	# group_execution.add_argument('--execution-providers', help = wording.get('execution_providers_help').format(choices = 'cpu'), dest = 'execution_providers', default = [ 'cpu' ], choices = encode_execution_providers(onnxruntime.get_available_providers()), nargs = '+')
	group_execution.add_argument('--execution-providers', help = wording.get('execution_providers_help').format(choices = 'cpu'), dest = 'execution_providers', default = [ 'cuda' ], choices = ["cuda"], nargs = '+')
	# group_execution.add_argument('--execution-thread-count', help = wording.get('execution_thread_count_help'), dest = 'execution_thread_count', type = int, default = 1)
	group_execution.add_argument('--execution-thread-count', help = wording.get('execution_thread_count_help'), dest = 'execution_thread_count', type = int, default = 8)
	group_execution.add_argument('--execution-queue-count', help = wording.get('execution_queue_count_help'), dest = 'execution_queue_count', type = int, default = 1)
	group_execution.add_argument('--max-memory', help=wording.get('max_memory_help'), dest='max_memory', type = int)
	# face recognition
	group_face_recognition = program.add_argument_group('face recognition')
	group_face_recognition.add_argument('--face-recognition', help = wording.get('face_recognition_help'), dest = 'face_recognition', default = 'reference', choices = facefusion.choices.face_recognitions)
	group_face_recognition.add_argument('--face-analyser-direction', help = wording.get('face_analyser_direction_help'), dest = 'face_analyser_direction', default = 'left-right', choices = facefusion.choices.face_analyser_directions)
	group_face_recognition.add_argument('--face-analyser-age', help = wording.get('face_analyser_age_help'), dest = 'face_analyser_age', choices = facefusion.choices.face_analyser_ages)
	group_face_recognition.add_argument('--face-analyser-gender', help = wording.get('face_analyser_gender_help'), dest = 'face_analyser_gender', choices = facefusion.choices.face_analyser_genders)
	group_face_recognition.add_argument('--reference-face-position', help = wording.get('reference_face_position_help'), dest = 'reference_face_position', type = int, default = 0)
	group_face_recognition.add_argument('--reference-face-distance', help = wording.get('reference_face_distance_help'), dest = 'reference_face_distance', type = float, default = 1.5)
	group_face_recognition.add_argument('--reference-frame-number', help = wording.get('reference_frame_number_help'), dest = 'reference_frame_number', type = int, default = 0)
	# frame extraction
	group_processing = program.add_argument_group('frame extraction')
	group_processing.add_argument('--trim-frame-start', help = wording.get('trim_frame_start_help'), dest = 'trim_frame_start', type = int)
	group_processing.add_argument('--trim-frame-end', help = wording.get('trim_frame_end_help'), dest = 'trim_frame_end', type = int)
	group_processing.add_argument('--temp-frame-format', help = wording.get('temp_frame_format_help'), dest = 'temp_frame_format', default = 'jpg', choices = facefusion.choices.temp_frame_formats)
	group_processing.add_argument('--temp-frame-quality', help = wording.get('temp_frame_quality_help'), dest = 'temp_frame_quality', type = int, default = 100, choices = range(101), metavar = '[0-100]')
	group_processing.add_argument('--keep-temp', help = wording.get('keep_temp_help'), dest = 'keep_temp', action = 'store_true')
	# output creation
	group_output = program.add_argument_group('output creation')
	group_output.add_argument('--output-image-quality', help=wording.get('output_image_quality_help'), dest = 'output_image_quality', type = int, default = 80, choices = range(101), metavar = '[0-100]')
	group_output.add_argument('--output-video-encoder', help = wording.get('output_video_encoder_help'), dest = 'output_video_encoder', default = 'libx264', choices = facefusion.choices.output_video_encoders)
	group_output.add_argument('--output-video-quality', help = wording.get('output_video_quality_help'), dest = 'output_video_quality', type = int, default = 80, choices = range(101), metavar = '[0-100]')
	group_output.add_argument('--keep-fps', help = wording.get('keep_fps_help'), dest = 'keep_fps', action = 'store_true')
	group_output.add_argument('--skip-audio', help = wording.get('skip_audio_help'), dest = 'skip_audio', action = 'store_true')
	# frame processors
	available_frame_processors = list_module_names('facefusion/processors/frame/modules')
	program = ArgumentParser(parents = [ program ], formatter_class = program.formatter_class, add_help = True)
	group_frame_processors = program.add_argument_group('frame processors')
	group_frame_processors.add_argument('--frame-processors', help = wording.get('frame_processors_help').format(choices = ', '.join(available_frame_processors)), dest = 'frame_processors', default = [ 'face_swapper' ], nargs = '+')
	for frame_processor in available_frame_processors:
		frame_processor_module = load_frame_processor_module(frame_processor)
		frame_processor_module.register_args(group_frame_processors)
	# uis
	group_uis = program.add_argument_group('uis')
	group_uis.add_argument('--ui-layouts', help = wording.get('ui_layouts_help').format(choices = ', '.join(list_module_names('facefusion/uis/layouts'))), dest = 'ui_layouts', default = [ 'default' ], nargs = '+')

	return program

def cli() -> None:
	run(get_argument_parser())

def apply_args(program : ArgumentParser) -> None:
	args = program.parse_args([])
	# general
	facefusion.globals.source_path = args.source_path
	facefusion.globals.target_path = args.target_path
	# facefusion.globals.output_path = normalize_output_path(facefusion.globals.source_path, facefusion.globals.target_path, args.output_path)
	_tmp_dir = Path(TEMP_DIRECTORY_PATH)
	_tmp_dir.mkdir(parents=True, exist_ok=True)
	facefusion.globals.output_path = str(_tmp_dir)
	# misc
	facefusion.globals.skip_download = args.skip_download
	facefusion.globals.headless = args.headless
	# execution
	facefusion.globals.execution_providers = decode_execution_providers(args.execution_providers)
	facefusion.globals.execution_thread_count = args.execution_thread_count
	facefusion.globals.execution_queue_count = args.execution_queue_count
	facefusion.globals.max_memory = args.max_memory
	# face recognition
	facefusion.globals.face_recognition = args.face_recognition
	facefusion.globals.face_analyser_direction = args.face_analyser_direction
	facefusion.globals.face_analyser_age = args.face_analyser_age
	facefusion.globals.face_analyser_gender = args.face_analyser_gender
	facefusion.globals.reference_face_position = args.reference_face_position
	facefusion.globals.reference_face_distance = args.reference_face_distance
	facefusion.globals.reference_frame_number = args.reference_frame_number
	# frame extraction
	facefusion.globals.trim_frame_start = args.trim_frame_start
	facefusion.globals.trim_frame_end = args.trim_frame_end
	facefusion.globals.temp_frame_format = args.temp_frame_format
	facefusion.globals.temp_frame_quality = args.temp_frame_quality
	facefusion.globals.keep_temp = args.keep_temp
	# output creation
	facefusion.globals.output_image_quality = args.output_image_quality
	facefusion.globals.output_video_encoder = args.output_video_encoder
	facefusion.globals.output_video_quality = args.output_video_quality
	facefusion.globals.keep_fps = args.keep_fps
	facefusion.globals.skip_audio = args.skip_audio
	# frame processors
	available_frame_processors = list_module_names('facefusion/processors/frame/modules')
	facefusion.globals.frame_processors = args.frame_processors
	for frame_processor in available_frame_processors:
		frame_processor_module = load_frame_processor_module(frame_processor)
		frame_processor_module.apply_args(program)
	# uis
	facefusion.globals.ui_layouts = args.ui_layouts


def run(program : ArgumentParser) -> None:
	apply_args(program)
	limit_resources()
	if not pre_check():
		return
	for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
		if not frame_processor_module.pre_check():
			return
	if facefusion.globals.headless:
		conditional_process()
	else:
		import facefusion.uis.core as ui

		for ui_layout in ui.get_ui_layouts_modules(facefusion.globals.ui_layouts):
			if not ui_layout.pre_check():
				return
		ui.launch()


def destroy() -> None:
	if facefusion.globals.target_path:
		clear_temp(facefusion.globals.target_path)
	sys.exit()


def limit_resources() -> None:
	# prevent tensorflow memory leak
	gpus = tensorflow.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
		tensorflow.config.experimental.set_virtual_device_configuration(gpu,
		[
			tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit = 512)
		])
	# limit memory usage
	if facefusion.globals.max_memory:
		memory = facefusion.globals.max_memory * 1024 ** 3
		if platform.system().lower() == 'darwin':
			memory = facefusion.globals.max_memory * 1024 ** 6
		if platform.system().lower() == 'windows':
			import ctypes
			kernel32 = ctypes.windll.kernel32 # type: ignore[attr-defined]
			kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
		else:
			import resource
			resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> bool:
	if sys.version_info < (3, 9):
		update_status(wording.get('python_not_supported').format(version = '3.9'))
		return False
	if not shutil.which('ffmpeg'):
		update_status(wording.get('ffmpeg_not_installed'))
		return False
	return True


def conditional_process(
	request: gradio.Request | None,
	task_id: str,
	width: int,
	height: int,
	output_path: str,
	preview_frame_slider: int,
	source_image: Frame,
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
) -> None:
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
		"reference_face_position_gallery_index": reference_face_position_gallery_index,
		"preview_frame_slider": preview_frame_slider,
	}
	for frame_processor_module in get_frame_processors_modules(frame_processors_checkbox_group):
		if not frame_processor_module.pre_process('output', kwargs):
			return
	if target_image is not None:
		process_image(
			request,
			width,
			height,
			output_path,
			frame_processors_checkbox_group,
			output_image_quality_slider,
			kwargs,
		)
	elif target_video is not None:
		process_video(
			request,
			task_id,
			width,
			height,
			output_path,
			frame_processors_checkbox_group,
			common_options_checkbox_group,
			temp_frame_format_dropdown,
			temp_frame_quality_slider,
			output_video_encoder_dropdown,
			output_video_quality_slider,
			trim_frame_start_slider,
			trim_frame_end_slider,
			kwargs,
		)

def process_image(
	request: gradio.Request | None,
	width: int,
	height: int,
	output_path: str,
	frame_processors_checkbox_group: list[str],
	output_image_quality_slider: int,
	kwargs: dict[str, Any],
) -> None:
	# if predict_image(facefusion.globals.target_path):
	# 	return
	frame = kwargs["target_image"]
	if predict_frame(frame):
		raise gradio.Error("The input image is NSFW, it is not supported now")
	# shutil.copy2(facefusion.globals.target_path, facefusion.globals.output_path)
	# process frame
	for frame_processor_module in get_frame_processors_modules(frame_processors_checkbox_group):
		update_status(wording.get('processing'), frame_processor_module.NAME)
		with monitor_call_context(
			request,
			"extensions.facefusion",
			"extensions.facefusion",
			decoded_params={
				"width": width,
				"height": height,
				"n_iter": 1,
			},
			only_available_for=["plus", "pro", "api"],
		):
			frame_processor_module.get_frame_processor(kwargs)
			frame = frame_processor_module.process_image_frame(frame, kwargs)
		frame_processor_module.post_process()

	write_image(output_path, frame)
	# compress image
	update_status(wording.get('compressing_image'))
	if not compress_image(output_path, output_image_quality_slider):
		update_status(wording.get('compressing_image_failed'))
	# validate image
	if is_image(output_path):
		update_status(wording.get('processing_image_succeed'))
	else:
		update_status(wording.get('processing_image_failed'))


def process_video(
	request: gradio.Request | None,
	task_id: str,
	width: int,
	height: int,
	output_path: str,
	frame_processors_checkbox_group: list[str],
	common_options_checkbox_group: list[str],
	temp_frame_format_dropdown: str,
	temp_frame_quality_slider: int,
	output_video_encoder_dropdown: str,
	output_video_quality_slider: int,
	trim_frame_start_slider: int,
	trim_frame_end_slider: int,
	kwargs: dict[str, Any],
) -> None:
	target_video = kwargs["target_video"]
	if predict_video(target_video):
		raise gradio.Error("The input video is NSFW, it is not supported now")
	fps = detect_fps(target_video) if "keep_fps" in common_options_checkbox_group else 25.0
	# create temp
	update_status(wording.get('creating_temp'))
	create_temp_dir(task_id)
	# extract frames
	update_status(wording.get('extracting_frames_fps').format(fps = fps))
	extract_frames(
		task_id,
		target_video,
		fps,
		temp_frame_format_dropdown,
		temp_frame_quality_slider,
		trim_frame_start_slider,
		trim_frame_end_slider,
	)
	# process frame
	temp_frame_paths = get_temp_video_frame_paths(task_id, temp_frame_format_dropdown)
	if temp_frame_paths:
		with monitor_call_context(
			request,
			"extensions.facefusion",
			"extensions.facefusion",
			decoded_params={
				"width": width,
				"height": height,
				"n_iter": len(temp_frame_paths) * len(frame_processors_checkbox_group),
			},
			only_available_for=["plus", "pro", "api"],
		):
			for frame_processor_module in get_frame_processors_modules(frame_processors_checkbox_group):
				update_status(wording.get('processing'), frame_processor_module.NAME)
				frame_processor_module.get_frame_processor(kwargs)
				frame_processor_module.process_video_frame(temp_frame_paths, kwargs)
				frame_processor_module.post_process()
	else:
		update_status(wording.get('temp_frames_not_found'))
		return
	# merge video
	update_status(wording.get('merging_video_fps').format(fps = fps))
	if not merge_video(
		task_id,
		fps,
		temp_frame_format_dropdown,
		output_video_encoder_dropdown,
		output_video_quality_slider,
	):
		update_status(wording.get('merging_video_failed'))
		return
	# handle audio
	if "skip_audio" in common_options_checkbox_group:
		update_status(wording.get('skipping_audio'))
		# move_temp(facefusion.globals.target_path, facefusion.globals.output_path)
		rename_temp_video_to_output(task_id, output_path)
	else:
		update_status(wording.get('restoring_audio'))
		if not restore_audio(
			task_id,
			fps,
			target_video,
			output_path,
			trim_frame_start_slider,
			trim_frame_end_slider,
		):
			update_status(wording.get('restoring_audio_failed'))
			# move_temp(facefusion.globals.target_path, facefusion.globals.output_path)
			rename_temp_video_to_output(task_id, output_path)
	# clear temp
	update_status(wording.get('clearing_temp'))
	# clear_temp(facefusion.globals.target_path)
	clear_temp_dir(task_id)

	# validate video
	# if is_video(facefusion.globals.target_path):
	if is_video(output_path):
		update_status(wording.get('processing_video_succeed'))
	else:
		update_status(wording.get('processing_video_failed'))


def update_status(message : str, scope : str = 'FACEFUSION.CORE') -> None:
	print('[' + scope + '] ' + message)
