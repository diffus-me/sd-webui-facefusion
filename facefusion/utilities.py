from typing import List, Optional
from functools import lru_cache
from pathlib import Path
from tqdm import tqdm
import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import tempfile
import urllib
import onnxruntime

import facefusion.globals
from facefusion import wording
from facefusion.vision import detect_fps
from modules.paths import models_path

_MODELS_DIR = os.path.join(models_path, "facefusion")

def update_model_path(models):
	for value in models.values():
		url = value["url"]
		filename = url.rsplit("/", 1)[-1]
		value["path"] = os.path.join(_MODELS_DIR, filename)

TEMP_DIRECTORY_PATH = os.path.join(tempfile.gettempdir(), 'facefusion')
TEMP_OUTPUT_VIDEO_NAME = 'temp.mp4'

# monkey patch ssl
if platform.system().lower() == 'darwin':
	ssl._create_default_https_context = ssl._create_unverified_context


def run_ffmpeg(args : List[str]) -> bool:
	commands = [ 'ffmpeg', '-hide_banner', '-loglevel', 'error' ]
	commands.extend(args)
	try:
		# subprocess.run(commands, stderr = subprocess.PIPE, check = True)
		subprocess.run(commands, capture_output=True, check = True)
		return True
	except subprocess.CalledProcessError:
		return False


def open_ffmpeg(args : List[str]) -> subprocess.Popen[bytes]:
	commands = [ 'ffmpeg', '-hide_banner', '-loglevel', 'error' ]
	commands.extend(args)
	return subprocess.Popen(commands, stdin = subprocess.PIPE)


def extract_frames(
	task_id: str,
	target_path: str,
	fps: float,
	temp_frame_format_dropdown: str,
	temp_frame_quality_slider: int,
	trim_frame_start_slider: int,
	trim_frame_end_slider: int,
) -> bool:
	temp_frame_compression = round(31 - (temp_frame_quality_slider * 0.31))
	trim_frame_start = trim_frame_start_slider
	trim_frame_end = trim_frame_end_slider
	# temp_frames_pattern = get_temp_frames_pattern(target_path, '%04d', temp_frame_format_dropdown)
	temp_frames_pattern = get_temp_video_frame_pattern(task_id, "%04d", temp_frame_format_dropdown)
	commands = [ '-hwaccel', 'auto', '-i', target_path, '-q:v', str(temp_frame_compression), '-pix_fmt', 'rgb24' ]
	if trim_frame_start is not None and trim_frame_end is not None:
		commands.extend([ '-vf', 'trim=start_frame=' + str(trim_frame_start) + ':end_frame=' + str(trim_frame_end) + ',fps=' + str(fps) ])
	elif trim_frame_start is not None:
		commands.extend([ '-vf', 'trim=start_frame=' + str(trim_frame_start) + ',fps=' + str(fps) ])
	elif trim_frame_end is not None:
		commands.extend([ '-vf', 'trim=end_frame=' + str(trim_frame_end) + ',fps=' + str(fps) ])
	else:
		commands.extend([ '-vf', 'fps=' + str(fps) ])
	commands.extend([ '-vsync', '0', temp_frames_pattern ])
	return run_ffmpeg(commands)


def compress_image(output_path : str, output_image_quality: int) -> bool:
	output_image_compression = round(31 - (output_image_quality * 0.31))
	commands = [ '-hwaccel', 'auto', '-i', output_path, '-q:v', str(output_image_compression), '-y', output_path ]
	return run_ffmpeg(commands)


def merge_video(
	task_id: str,
	fps: float,
	temp_frame_format_dropdown: str,
	output_video_encoder_dropdown: str,
	output_video_quality_slider: int,
) -> bool:
	# temp_output_video_path = get_temp_output_video_path(target_path)
	# temp_frames_pattern = get_temp_frames_pattern(target_path, '%04d',  temp_frame_format_dropdown)
	temp_output_video_path = str(get_temp_video_path(task_id))
	temp_frames_pattern = get_temp_video_frame_pattern(task_id, "%04d",  temp_frame_format_dropdown)
	commands = [ '-hwaccel', 'auto', '-r', str(fps), '-i', temp_frames_pattern, '-c:v', output_video_encoder_dropdown ]
	if output_video_encoder_dropdown in [ 'libx264', 'libx265' ]:
		output_video_compression = round(51 - (output_video_quality_slider * 0.51))
		commands.extend([ '-crf', str(output_video_compression) ])
	if output_video_encoder_dropdown in [ 'libvpx-vp9' ]:
		output_video_compression = round(63 - (output_video_quality_slider * 0.63))
		commands.extend([ '-crf', str(output_video_compression) ])
	if output_video_encoder_dropdown in [ 'h264_nvenc', 'hevc_nvenc' ]:
		output_video_compression = round(51 - (output_video_quality_slider * 0.51))
		commands.extend([ '-cq', str(output_video_compression) ])
	commands.extend([ '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-y', temp_output_video_path ])
	return run_ffmpeg(commands)


def restore_audio(
	task_id: str,
	fps: float,
	target_path: str,
	output_path: str,
	trim_frame_start_slider: int,
	trim_frame_end_slider: int,
) -> bool:
	# fps = detect_fps(target_path)
	trim_frame_start = trim_frame_start_slider
	trim_frame_end = trim_frame_end_slider
	# temp_output_video_path = get_temp_output_video_path(target_path)
	temp_output_video_path = str(get_temp_video_path(task_id))
	commands = [ '-hwaccel', 'auto', '-i', temp_output_video_path ]
	if trim_frame_start is not None:
		start_time = trim_frame_start / fps
		commands.extend([ '-ss', str(start_time) ])
	if trim_frame_end is not None:
		end_time = trim_frame_end / fps
		commands.extend([ '-to', str(end_time) ])
	commands.extend([ '-i', target_path, '-c',  'copy', '-map', '0:v:0', '-map', '1:a:0', '-shortest', '-y', output_path ])
	return run_ffmpeg(commands)


def get_temp_frame_paths(target_path : str, temp_frame_format_dropdown: str) -> List[str]:
	temp_frames_pattern = get_temp_frames_pattern(target_path, '*', temp_frame_format_dropdown)
	return sorted(glob.glob(temp_frames_pattern))


def get_temp_frames_pattern(target_path : str, temp_frame_prefix : str, temp_frame_format_dropdown: str) -> str:
	temp_directory_path = get_temp_directory_path(target_path)
	return os.path.join(temp_directory_path, temp_frame_prefix + '.' + temp_frame_format_dropdown)


def get_temp_directory_path(target_path : str) -> str:
	target_name, _ = os.path.splitext(os.path.basename(target_path))
	return os.path.join(TEMP_DIRECTORY_PATH, target_name)


def get_temp_output_video_path(target_path : str) -> str:
	temp_directory_path = get_temp_directory_path(target_path)
	return os.path.join(temp_directory_path, TEMP_OUTPUT_VIDEO_NAME)


def normalize_output_path(source_path : Optional[str], target_path : Optional[str], output_path : Optional[str]) -> Optional[str]:
	if is_file(source_path) and is_file(target_path) and is_directory(output_path):
		source_name, _ = os.path.splitext(os.path.basename(source_path))
		target_name, target_extension = os.path.splitext(os.path.basename(target_path))
		return os.path.join(output_path, source_name + '-' + target_name + target_extension)
	if is_file(target_path) and output_path:
		target_name, target_extension = os.path.splitext(os.path.basename(target_path))
		output_name, output_extension = os.path.splitext(os.path.basename(output_path))
		output_directory_path = os.path.dirname(output_path)
		if is_directory(output_directory_path) and output_extension:
			return os.path.join(output_directory_path, output_name + target_extension)
		return None
	return output_path


def create_temp(target_path : str) -> None:
	temp_directory_path = get_temp_directory_path(target_path)
	Path(temp_directory_path).mkdir(parents = True, exist_ok = True)


def move_temp(target_path : str, output_path : str) -> None:
	temp_output_video_path = get_temp_output_video_path(target_path)
	if is_file(temp_output_video_path):
		if is_file(output_path):
			os.remove(output_path)
		shutil.move(temp_output_video_path, output_path)


def clear_temp(target_path : str) -> None:
	temp_directory_path = get_temp_directory_path(target_path)
	parent_directory_path = os.path.dirname(temp_directory_path)
	if not facefusion.globals.keep_temp and is_directory(temp_directory_path):
		shutil.rmtree(temp_directory_path)
	if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
		os.rmdir(parent_directory_path)


def get_work_dir(task_id: str) -> Path:
	return Path(TEMP_DIRECTORY_PATH) / task_id

def create_work_dir(task_id: str) -> None:
	work_dir = get_work_dir(task_id)
	work_dir.mkdir(parents=True, exist_ok=True)


def get_output_path(task_id: str, extension: str) -> Path:
	return (get_work_dir(task_id) / "output").with_suffix(extension)

def get_temp_dir(task_id: str) -> Path:
	return get_work_dir(task_id) / "temp"

def create_temp_dir(task_id: str) -> None:
	temp_dir = get_temp_dir(task_id)
	temp_dir.mkdir(parents=True, exist_ok=True)

def get_temp_video_path(task_id: str) -> Path:
	work_dir = get_work_dir(task_id)
	return work_dir / TEMP_OUTPUT_VIDEO_NAME

def get_temp_video_frame_pattern(task_id: str, stem: str, temp_frame_format_dropdown: str) -> str:
	temp_dir = get_temp_dir(task_id)
	return str((temp_dir / stem).with_suffix(f".{temp_frame_format_dropdown}"))

def get_temp_video_frame_paths(task_id: str, temp_frame_format_dropdown: str) -> list[str]:
	pattern = get_temp_video_frame_pattern(task_id, "*", temp_frame_format_dropdown)
	return sorted(glob.glob(pattern))

def rename_temp_video_to_output(task_id: str, output_path: str) -> None:
	temp_video_path = get_temp_video_path(task_id)
	temp_video_path.rename(output_path)

def clear_temp_dir(task_id: str) -> None:
	temp_dir = get_temp_dir(task_id)
	shutil.rmtree(temp_dir)




def is_file(file_path : str) -> bool:
	return bool(file_path and os.path.isfile(file_path))


def is_directory(directory_path : str) -> bool:
	return bool(directory_path and os.path.isdir(directory_path))


def is_image(image_path : str) -> bool:
	if is_file(image_path):
		mimetype, _ = mimetypes.guess_type(image_path)
		return bool(mimetype and mimetype.startswith('image/'))
	return False


def is_video(video_path : str) -> bool:
	if is_file(video_path):
		mimetype, _ = mimetypes.guess_type(video_path)
		return bool(mimetype and mimetype.startswith('video/'))
	return False


def conditional_download(download_directory_path : str, urls : List[str]) -> None:
	for url in urls:
		download_file_path = os.path.join(download_directory_path, os.path.basename(url))
		total = get_download_size(url)
		if is_file(download_file_path):
			initial = os.path.getsize(download_file_path)
		else:
			initial = 0
		if initial < total:
			with tqdm(total = total, initial = initial, desc = wording.get('downloading'), unit = 'B', unit_scale = True, unit_divisor = 1024) as progress:
				subprocess.Popen([ 'curl', '--create-dirs', '--silent', '--insecure', '--location', '--continue-at', '-', '--output', download_file_path, url ])
				current = initial
				while current < total:
					if is_file(download_file_path):
						current = os.path.getsize(download_file_path)
						progress.update(current - progress.n)


@lru_cache(maxsize = None)
def get_download_size(url : str) -> int:
	try:
		response = urllib.request.urlopen(url) # type: ignore[attr-defined]
		return int(response.getheader('Content-Length'))
	except (OSError, ValueError):
		return 0


def is_download_done(url : str, file_path : str) -> bool:
	if is_file(file_path):
		return get_download_size(url) == os.path.getsize(file_path)
	return False


def resolve_relative_path(path : str) -> str:
	return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


_PROJECT_DIR = Path(__file__).absolute().parent.parent

def list_module_names(path : str) -> Optional[List[str]]:
	path = str(_PROJECT_DIR / path)
	if os.path.exists(path):
		files = os.listdir(path)
		return [ Path(file).stem for file in files if not Path(file).stem.startswith('__') ]
	return None


def encode_execution_providers(execution_providers : List[str]) -> List[str]:
	return [ execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers ]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
	available_execution_providers = onnxruntime.get_available_providers()
	encoded_execution_providers = encode_execution_providers(available_execution_providers)
	return [ execution_provider for execution_provider, encoded_execution_provider in zip(available_execution_providers, encoded_execution_providers) if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers) ]


def get_device(execution_providers : List[str]) -> str:
	if 'CUDAExecutionProvider' in execution_providers:
		return 'cuda'
	if 'CoreMLExecutionProvider' in execution_providers:
		return 'mps'
	return 'cpu'
