#!/usr/bin/env python3

import gradio as gr
from modules import script_callbacks

from facefusion.core import apply_args, get_argument_parser, limit_resources, pre_check
from facefusion.processors.frame.modules import face_enhancer, face_swapper, frame_enhancer
from facefusion.uis.layouts import default


def on_ui_tabs():
    apply_args(get_argument_parser())
    limit_resources()

    if not pre_check():
        return

    if (
        not face_enhancer.pre_check()
        or not face_swapper.pre_check()
        or not frame_enhancer.pre_check()
    ):
        return

    if not default.pre_check():
        return

    with gr.Blocks() as block:
        if default.pre_render():
            default.render()
            default.listen()

        return ((block, "Video Face Swapper (FaceFusion)", "facefusion"),)


script_callbacks.on_ui_tabs(on_ui_tabs)
