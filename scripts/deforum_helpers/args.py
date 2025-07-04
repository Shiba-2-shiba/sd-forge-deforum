# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

import json
import os
import pathlib
import tempfile
import time
from types import SimpleNamespace

import modules.paths as ph
import modules.shared as sh
from PIL import Image
from modules.processing import get_fixed_seed

from .defaults import (get_guided_imgs_default_json, get_camera_shake_list, get_keyframe_distribution_list,
                       get_samplers_list, get_schedulers_list)
from .deforum_controlnet import controlnet_component_names
from .general_utils import get_os, substitute_placeholders


def RootArgs():
    return {
        "device": sh.device,
        "models_path": ph.models_path + '/Deforum',
        "half_precision": not sh.cmd_opts.no_half,
        "clipseg_model": None,
        "mask_preset_names": ['everywhere', 'video_mask'],
        "frames_cache": [],
        "raw_batch_name": None,
        "raw_seed": None,
        "timestring": "",
        "subseed": -1,
        "subseed_strength": 0,
        "seed_internal": 0,
        "init_sample": None,
        "noise_mask": None,
        "initial_info": None,
        "first_frame": None,
        "animation_prompts": None,
        "prompt_keyframes": None,
        "current_user_os": get_os(),
        "tmp_deforum_run_duplicated_folder": os.path.join(tempfile.gettempdir(), 'tmp_run_deforum')
    }


# 'Midas-3.1-BeitLarge' is temporarily removed until fixed. Can add it back anytime
# as it's supported in the back-end depth code
def DeforumAnimArgs():
    return {
        "animation_mode": {
            "label": "Animation mode",
            "type": "radio",
            "choices": ['2D', '3D', 'Video Input', 'Interpolation', 'Wan Video', 'FramePack F1'],
            "value": "2D",
            "info": "control animation mode, will hide non relevant params upon change"
        },
        "max_frames": {
            "label": "Max frames",
            "type": "number",
            "precision": 0,
            "value": 333,
            "info": "end the animation at this frame number",
        },
        "border": {
            "label": "Border mode",
            "type": "radio",
            "choices": ['replicate', 'wrap'],
            "value": "replicate",
            "info": "controls pixel generation method for images smaller than the frame. hover on the options to see more info"
        },
        "angle": {
            "label": "Angle",
            "type": "textbox",
            "value": "0: (0)",
            "info": "rotate canvas clockwise/anticlockwise in degrees per frame"
        },
        "zoom": {
            "label": "Zoom",
            "type": "textbox",
            "value": "0: (1.0)",  # original value: "0: (1.0025+0.002*sin(1.25*3.14*t/120))"
            "info": "scale the canvas size, multiplicatively. [static = 1.0]"
        },
        "translation_x": {
            "label": "Translation X",
            "type": "textbox",
            "value": "0: (0)",
            "info": "move canvas left/right in pixels per frame"
        },
        "translation_y": {
            "label": "Translation Y",
            "type": "textbox",
            "value": "0: (0)",
            "info": "move canvas up/down in pixels per frame"
        },
        "translation_z": {
            "label": "Translation Z (zoom when animation mode is '3D')",
            "type": "textbox",
            "value": "0: (0)",  # original value: "0: (1.10)"
            "info": "move canvas towards/away from view [speed set by FOV]"
        },
        "transform_center_x": {
            "label": "Transform Center X",
            "type": "textbox",
            "value": "0: (0.5)",
            "info": "X center axis for 2D angle/zoom"
        },
        "transform_center_y": {
            "label": "Transform Center Y",
            "type": "textbox",
            "value": "0: (0.5)",
            "info": "Y center axis for 2D angle/zoom"
        },
        "rotation_3d_x": {
            "label": "Rotation 3D X",
            "type": "textbox",
            "value": "0: (0)",
            "info": "tilt canvas up/down in degrees per frame"
        },
        "rotation_3d_y": {
            "label": "Rotation 3D Y",
            "type": "textbox",
            "value": "0: (0)",
            "info": "pan canvas left/right in degrees per frame"
        },
        "rotation_3d_z": {
            "label": "Rotation 3D Z",
            "type": "textbox",
            "value": "0: (0)",
            "info": "roll canvas clockwise/anticlockwise"
        },
        "shake_name": {
            "label": "Shake Name",
            "type": "dropdown",
            "choices": get_camera_shake_list().values(),
            "value": "INVESTIGATION",
            "info": "Name of the camera shake loop.",
        },
        "shake_intensity": {
            "label": "Shake Intensity",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 3.0,
            "step": 0.1,
            "value": 1.0,
            "info": "Intensity of the camera shake loop."
        },
        "shake_speed": {
            "label": "Shake Speed",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 3.0,
            "step": 0.1,
            "value": 1.0,
            "info": "Speed of the camera shake loop."
        },
        "enable_perspective_flip": {
            "label": "Enable perspective flip",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "perspective_flip_theta": {
            "label": "Perspective flip theta",
            "type": "textbox",
            "value": "0: (0)",
            "info": ""
        },
        "perspective_flip_phi": {
            "label": "Perspective flip phi",
            "type": "textbox",
            "value": "0: (0)",
            "info": ""
        },
        "perspective_flip_gamma": {
            "label": "Perspective flip gamma",
            "type": "textbox",
            "value": "0: (0)",
            "info": ""
        },
        "perspective_flip_fv": {
            "label": "Perspective flip tv",
            "type": "textbox",
            "value": "0: (53)",
            "info": "the 2D vanishing point of perspective (rec. range 30-160)"
        },
        "noise_schedule": {
            "label": "Noise schedule",
            "type": "textbox",
            "value": "0: (0.065)",
            "info": ""
        },
        "strength_schedule": {
            "label": "Strength schedule",
            "type": "textbox",
            "value": "0: (0.85)",
            "info": "amount of presence of previous frame to influence next frame, also controls steps in the following formula [steps - (strength_schedule * steps)]"
        },
        "keyframe_strength_schedule": {
            "label": "Strength schedule for keyframes",
            "type": "textbox",
            "value": "0: (0.50)",
            "info": "like 'Strength schedule' but only for frames with an entry in 'prompts'. Meant to be set somewhat lower than the regular Strengh schedule. At 0 it generates a totally new image on every prompt change. Ignored if Parseq is used or when Keyframe distribustion is disabled."
        },
        "contrast_schedule": "0: (1.0)",
        "cfg_scale_schedule": {
            "label": "CFG scale schedule",
            "type": "textbox",
            "value": "0: (1.0)",
            "info": "how closely the image should conform to the prompt. Lower values produce more creative results. (recommended value for Flux.1: 1.0, 5-15 with other models)"
        },
        "distilled_cfg_scale_schedule": {
            "label": "Distilled CFG scale schedule",
            "type": "textbox",
            "value": "0: (3.5)",
            "info": "how closely the image should conform to the prompt. Lower values produce more creative results. (recommended value for Flux.1: 3.5, ignored for most other models)`"
        },
        "enable_steps_scheduling": {
            "label": "Enable steps scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "steps_schedule": {
            "label": "Steps schedule",
            "type": "textbox",
            "value": "0: (20)",
            "info": "mainly allows using more than 200 steps. Otherwise, it's a mirror-like param of 'strength schedule'"
        },
        "fov_schedule": {
            "label": "FOV schedule",
            "type": "textbox",
            "value": "0: (70)",
            "info": "adjusts the scale at which the canvas is moved in 3D by the translation_z value. [Range -180 to +180, with 0 being undefined. Values closer to 180 will make the image have less depth, while values closer to 0 will allow more depth]"
        },
        "aspect_ratio_schedule": {
            "label": "Aspect Ratio schedule",
            "type": "textbox",
            "value": "0: (1.0)",
            "info": "adjusts the aspect ratio for the depth calculations. HAS NOTHING TO DO WITH THE VIDEO SIZE. Meant to be left at 1.0, except for emulating legacy behaviour."
        },
        "aspect_ratio_use_old_formula": {
            "label": "Use old aspect ratio formula",
            "type": "checkbox",
            "value": False,
            "info": "for backward compatibility. Uses the formula: `width/height`"
        },
        "near_schedule": {
            "label": "Near schedule",
            "type": "textbox",
            "value": "0: (200)",
            "info": ""
        },
        "far_schedule": {
            "label": "Far schedule",
            "type": "textbox",
            "value": "0: (10000)",
            "info": ""
        },
        "seed_schedule": {
            "label": "Seed schedule",
            "type": "textbox",
            "value": '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)',
            "info": ""
        },
        "enable_subseed_scheduling": {
            "label": "Enable Subseed scheduling",
            "type": "checkbox",
            "value": False,
            "info": "Aim for more continuous similarity by mixing in a constant seed."
        },
        "subseed_schedule": {
            "label": "Subseed schedule",
            "type": "textbox",
            "value": "0: (1)",
            "info": ""
        },
        "subseed_strength_schedule": {
            "label": "Subseed strength schedule",
            "type": "textbox",
            "value": "0: (0)",
            "info": ""
        },
        "enable_sampler_scheduling": {
            "label": "Enable sampler scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "sampler_schedule": {
            "label": "Sampler schedule",
            "type": "textbox",
            "value": '0: ("Euler")',
            "info": "allows keyframing of samplers. Use names as they appear in ui dropdown in 'run' tab"
        },
        "enable_scheduler_scheduling": {
            "label": "Enable scheduler scheduling",
            "type": "checkbox",
            "value": False,
            "info": "enables scheduling of scheduler schedules."
        },
        "scheduler_schedule": {
            "label": "Scheduler schedule",
            "type": "textbox",
            "value": '0: ("Simple")',
            "info": "allows keyframing of schedulers. Use names as they appear in ui dropdown in 'run' tab"
        },
        "use_noise_mask": {
            "label": "Use noise mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "mask_schedule": {
            "label": "Mask schedule",
            "type": "textbox",
            "value": '0: ("{video_mask}")',
            "info": ""
        },
        "noise_mask_schedule": {
            "label": "Noise mask schedule",
            "type": "textbox",
            "value": '0: ("{video_mask}")',
            "info": ""
        },
        "enable_checkpoint_scheduling": {
            "label": "Enable checkpoint scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "checkpoint_schedule": {
            "label": "allows keyframing different sd models. Use *full* name as appears in ui dropdown",
            "type": "textbox",
            "value": '0: ("model1.ckpt"), 100: ("model2.safetensors")',
            "info": "allows keyframing different sd models. Use *full* name as appears in ui dropdown"
        },
        "enable_clipskip_scheduling": {
            "label": "Enable CLIP skip scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "clipskip_schedule": {
            "label": "CLIP skip schedule",
            "type": "textbox",
            "value": "0: (2)",
            "info": ""
        },
        "enable_noise_multiplier_scheduling": {
            "label": "Enable noise multiplier scheduling",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "noise_multiplier_schedule": {
            "label": "Noise multiplier schedule",
            "type": "textbox",
            "value": "0: (1.05)",
            "info": ""
        },
        "resume_from_timestring": {
            "label": "Resume from timestring",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "resume_timestring": {
            "label": "Resume timestring",
            "type": "textbox",
            "value": "20241111111111",
            "info": ""
        },
        "enable_ddim_eta_scheduling": {
            "label": "Enable DDIM ETA scheduling",
            "type": "checkbox",
            "value": False,
            "visible": False,
            "info": "noise multiplier; higher = more unpredictable results"
        },
        "ddim_eta_schedule": {
            "label": "DDIM ETA Schedule",
            "type": "textbox",
            "value": "0: (0)",
            "visible": False,
            "info": ""
        },
        "enable_ancestral_eta_scheduling": {
            "label": "Enable Ancestral ETA scheduling",
            "type": "checkbox",
            "value": False,
            "info": "noise multiplier; applies to Euler A and other samplers that have the letter 'a' in them"
        },
        "ancestral_eta_schedule": {
            "label": "Ancestral ETA Schedule",
            "type": "textbox",
            "value": "0: (1)",
            "visible": False,
            "info": ""
        },
        "amount_schedule": {
            "label": "Amount schedule",
            "type": "textbox",
            "value": "0: (0.1)",
            "info": ""
        },
        "kernel_schedule": {
            "label": "Kernel schedule",
            "type": "textbox",
            "value": "0: (5)",
            "info": ""
        },
        "sigma_schedule": {
            "label": "Sigma schedule",
            "type": "textbox",
            "value": "0: (1)",
            "info": ""
        },
        "threshold_schedule": {
            "label": "Threshold schedule",
            "type": "textbox",
            "value": "0: (0)",
            "info": ""
        },
        "color_coherence": {
            "label": "Color coherence",
            "type": "dropdown",
            "choices": ['None', 'HSV', 'LAB', 'RGB', 'Video Input', 'Image'],
            "value": "None",
            "info": "choose an algorithm/ method for keeping color coherence across the animation"
        },
        "color_coherence_image_path": {
            "label": "Color coherence image path",
            "type": "textbox",
            "value": "",
            "info": ""
        },
        "color_coherence_video_every_N_frames": {
            "label": "Color coherence video every N frames",
            "type": "number",
            "precision": 0,
            "value": 1,
            "info": "",
        },
        "color_force_grayscale": {
            "label": "Color force Grayscale",
            "type": "checkbox",
            "value": False,
            "info": "force all frames to be in grayscale"
        },
        "legacy_colormatch": {
            "label": "Legacy colormatch",
            "type": "checkbox",
            "value": False,
            "info": "apply colormatch before adding noise (use with CN's Tile)"
        },
        "keyframe_distribution": {
            "label": "Keyframe distribution.",
            "type": "dropdown",
            "choices": get_keyframe_distribution_list().values(),
            "value": "Keyframes Only",
            "info": "Allows for fast generations at high cadence or no cadence."
        },
        "diffusion_cadence": {
            "label": "Cadence",
            "type": "slider",
            "minimum": 1,
            "maximum": 200,
            "step": 1,
            "value": 10,
            "info": "# of in-between frames that will not be directly diffused"
        },
        "optical_flow_cadence": {
            "label": "Optical flow cadence",
            "type": "dropdown",
            "choices": ['None', 'RAFT', 'DIS Medium', 'DIS Fine', 'Farneback'],
            "value": "None",
            "info": "use optical flow estimation for your in-between (cadence) frames"
        },
        "cadence_flow_factor_schedule": {
            "label": "Cadence flow factor schedule",
            "type": "textbox",
            "value": "0: (1)",
            "info": ""
        },
        "optical_flow_redo_generation": {
            "label": "Optical flow generation",
            "type": "dropdown",
            "choices": ['None', 'RAFT', 'DIS Medium', 'DIS Fine', 'Farneback'],
            "value": "None",
            "info": "this option takes twice as long because it generates twice in order to capture the optical flow from the previous image to the first generation, then warps the previous image and redoes the generation"
        },
        "redo_flow_factor_schedule": {
            "label": "Generation flow factor schedule",
            "type": "textbox",
            "value": "0: (1)",
            "info": ""
        },
        "diffusion_redo": '0',
        "noise_type": {
            "label": "Noise type",
            "type": "radio",
            "choices": ['uniform', 'perlin'],
            "value": "perlin",
            "info": ""
        },
        "perlin_w": {
            "label": "Perlin W",
            "type": "slider",
            "minimum": 0.1,
            "maximum": 16,
            "step": 0.1,
            "value": 8,
            "visible": False
        },
        "perlin_h": {
            "label": "Perlin H",
            "type": "slider",
            "minimum": 0.1,
            "maximum": 16,
            "step": 0.1,
            "value": 8,
            "visible": False
        },
        "perlin_octaves": {
            "label": "Perlin octaves",
            "type": "slider",
            "minimum": 1,
            "maximum": 7,
            "step": 1,
            "value": 4
        },
        "perlin_persistence": {
            "label": "Perlin persistence",
            "type": "slider",
            "minimum": 0,
            "maximum": 1,
            "step": 0.02,
            "value": 0.5
        },
        "use_depth_warping": {
            "label": "Use depth warping",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "depth_algorithm": {
            "label": "Depth Algorithm",
            "type": "dropdown",
            "choices": ['Depth-Anything-V2-Small', 'Midas-3-Hybrid', 'Midas+AdaBins (old)', 'Zoe+AdaBins (old)', 'AdaBins', 'Zoe', 'Leres'],
            "value": "Depth-Anything-V2-Small",
            "info": "choose an algorithm/ method for keeping color coherence across the animation"
        },
        "midas_weight": {
            "label": "MiDaS/Zoe weight",
            "type": "number",
            "precision": None,
            "value": 0.2,
            "info": "sets a midpoint at which a depth-map is to be drawn: range [-1 to +1]",
            "visible": False
        },
        "padding_mode": {
            "label": "Padding mode",
            "type": "radio",
            "choices": ['border', 'reflection', 'zeros'],
            "value": "border",
            "info": "Choose how to handle pixels outside the field of view as they come into the scene"
        },
        "sampling_mode": {
            "label": "Sampling mode",
            "type": "radio",
            "choices": ['bicubic', 'bilinear', 'nearest'],
            "value": "bicubic",
            "info": "Choose the sampling method: Bicubic for quality, Bilinear for speed, Nearest for simplicity."
        },
        "save_depth_maps": {
            "label": "Save 3D depth maps",
            "type": "checkbox",
            "value": False,
            "info": "save animation's depth maps as extra files"
        },
        "video_init_path": {
            "label": "Video init path/ URL",
            "type": "textbox",
            "value": 'https://deforum.github.io/a1/V1.mp4',
            "info": ""
        },
        "extract_nth_frame": {
            "label": "Extract nth frame",
            "type": "number",
            "precision": 0,
            "value": 1,
            "info": ""
        },
        "extract_from_frame": {
            "label": "Extract from frame",
            "type": "number",
            "precision": 0,
            "value": 0,
            "info": ""
        },
        "extract_to_frame": {
            "label": "Extract to frame",
            "type": "number",
            "precision": 0,
            "value": -1,
            "info": ""
        },
        "overwrite_extracted_frames": {
            "label": "Overwrite extracted frames",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "use_mask_video": {
            "label": "Use mask video",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "video_mask_path": {
            "label": "Video mask path",
            "type": "textbox",
            "value": 'https://deforum.github.io/a1/VM1.mp4',
            "info": ""
        },
        "hybrid_comp_alpha_schedule": {
            "label": "Comp alpha schedule",
            "type": "textbox",
            "value": "0:(0.5)",
            "info": ""
        },
        "hybrid_comp_mask_blend_alpha_schedule": {
            "label": "Comp mask blend alpha schedule",
            "type": "textbox",
            "value": "0:(0.5)",
            "info": ""
        },
        "hybrid_comp_mask_contrast_schedule": {
            "label": "Comp mask contrast schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": ""
        },
        "hybrid_comp_mask_auto_contrast_cutoff_high_schedule": {
            "label": "Comp mask auto contrast cutoff high schedule",
            "type": "textbox",
            "value": "0:(100)",
            "info": ""
        },
        "hybrid_comp_mask_auto_contrast_cutoff_low_schedule": {
            "label": "Comp mask auto contrast cutoff low schedule",
            "type": "textbox",
            "value": "0:(0)",
            "info": ""
        },
        "hybrid_flow_factor_schedule": {
            "label": "Flow factor schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": ""
        },
        "hybrid_generate_inputframes": {
            "label": "Generate inputframes",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "hybrid_generate_human_masks": {
            "label": "Generate human masks",
            "type": "radio",
            "choices": ['None', 'PNGs', 'Video', 'Both'],
            "value": "None",
            "info": ""
        },
        "hybrid_use_first_frame_as_init_image": {
            "label": "First frame as init image",
            "type": "checkbox",
            "value": True,
            "info": "",
            "visible": False
        },
        "hybrid_motion": {
            "label": "Hybrid motion",
            "type": "radio",
            "choices": ['None', 'Optical Flow', 'Perspective', 'Affine'],
            "value": "None",
            "info": ""
        },
        "hybrid_motion_use_prev_img": {
            "label": "Motion use prev img",
            "type": "checkbox",
            "value": False,
            "info": "",
            "visible": False
        },
        "hybrid_flow_consistency": {
            "label": "Flow consistency mask",
            "type": "checkbox",
            "value": False,
            "info": "",
            "visible": False
        },
        "hybrid_consistency_blur": {
            "label": "Consistency mask blur",
            "type": "slider",
            "minimum": 0,
            "maximum": 16,
            "step": 1,
            "value": 2,
            "visible": False
        },
        "hybrid_flow_method": {
            "label": "Flow method",
            "type": "radio",
            "choices": ['RAFT', 'DIS Medium', 'DIS Fine', 'Farneback'],
            "value": "RAFT",
            "info": "",
            "visible": False
        },
        "hybrid_composite": 'None',  # ['None', 'Normal', 'Before Motion', 'After Generation']
        "hybrid_use_init_image": {
            "label": "Use init image as video",
            "type": "checkbox",
            "value": False,
            "info": "",
        },
        "hybrid_comp_mask_type": {
            "label": "Comp mask type",
            "type": "radio",
            "choices": ['None', 'Depth', 'Video Depth', 'Blend', 'Difference'],
            "value": "None",
            "info": "",
            "visible": False
        },
        "hybrid_comp_mask_inverse": False,
        "hybrid_comp_mask_equalize": {
            "label": "Comp mask equalize",
            "type": "radio",
            "choices": ['None', 'Before', 'After', 'Both'],
            "value": "None",
            "info": "",
        },
        "hybrid_comp_mask_auto_contrast": False,
        "hybrid_comp_save_extra_frames": False
    }


def DeforumArgs():
    return {
        "W": {
            "label": "Width",
            "type": "slider",
            "minimum": 64,
            "maximum": 2048,
            "step": 4,
            "value": 1280,
        },
        "H": {
            "label": "Height",
            "type": "slider",
            "minimum": 64,
            "maximum": 2048,
            "step": 4,
            "value": 720,
        },
        "show_info_on_ui": True,
        "tiling": {
            "label": "Tiling",
            "type": "checkbox",
            "value": False,
            "info": "enable for seamless-tiling of each generated image. Experimental"
        },
        "restore_faces": {
            "label": "Restore faces",
            "type": "checkbox",
            "value": False,
            "info": "enable to trigger webui's face restoration on each frame during the generation"
        },
        "seed_resize_from_w": {
            "label": "Resize seed from width",
            "type": "slider",
            "minimum": 0,
            "maximum": 2048,
            "step": 64,
            "value": 0,
        },
        "seed_resize_from_h": {
            "label": "Resize seed from height",
            "type": "slider",
            "minimum": 0,
            "maximum": 2048,
            "step": 64,
            "value": 0,
        },
        "seed": {
            "label": "Seed",
            "type": "number",
            "precision": 0,
            "value": -1,
            "info": "Starting seed for the animation. -1 for random"
        },
        "sampler": {
            "label": "Sampler",
            "type": "dropdown",
            "choices": get_samplers_list().values(),
            "value": "Euler",
        },
        "scheduler": {
            "label": "Scheduler",
            "type": "dropdown",
            "choices": get_schedulers_list().values(),
            "value": "Simple",
        },
        "steps": {
            "label": "Steps",
            "type": "slider",
            "minimum": 1,
            "maximum": 200,
            "step": 1,
            "value": 20,
        },
        "batch_name": {
            "label": "Batch name",
            "type": "textbox",
            "value": "Deforum_{timestring}",
            "info": "output images will be placed in a folder with this name ({timestring} token will be replaced) inside the img2img output folder. Supports params placeholders. e.g {seed}, {w}, {h}, {prompts}"
        },
        "seed_behavior": {
            "label": "Seed behavior",
            "type": "radio",
            "choices": ['iter', 'fixed', 'random', 'ladder', 'alternate', 'schedule'],
            "value": "iter",
            "info": "controls the seed behavior that is used for animation. Hover on the options to see more info"
        },
        "seed_iter_N": {
            "label": "Seed iter N",
            "type": "number",
            "precision": 0,
            "value": 1,
            "info": "for how many frames the same seed should stick before iterating to the next one"
        },
        "use_init": {
            "label": "Use init",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "strength": {
            "label": "Strength",
            "type": "slider",
            "minimum": 0,
            "maximum": 1,
            "step": 0.01,
            "value": 0.85,
            "info": "the inverse of denoise; lower values alter the init image more (high denoise); higher values alter it less (low denoise)"
        },
        "strength_0_no_init": {
            "label": "Strength 0 no init",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "init_image": {
            "label": "Init image URL",
            "type": "textbox",
            "value": "https://deforum.github.io/a1/I1.png",
            "info": "Use web address or local path. Note: if the image box below is used then this field is ignored."
        },
        "init_image_box": {
            "label": "Init image box",
            "type": "image",
            "type_param": "pil",
            "source": "upload",
            "interactive": True,
            "info": ""
        },
        "use_mask": {
            "label": "Use mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "use_alpha_as_mask": {
            "label": "Use alpha as mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "mask_file": {
            "label": "Mask file",
            "type": "textbox",
            "value": "https://deforum.github.io/a1/M1.jpg",
            "info": ""
        },
        "invert_mask": {
            "label": "Invert mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "mask_contrast_adjust": {
            "label": "Mask contrast adjust",
            "type": "number",
            "precision": None,
            "value": 1.0,
            "info": ""
        },
        "mask_brightness_adjust": {
            "label": "Mask brightness adjust",
            "type": "number",
            "precision": None,
            "value": 1.0,
            "info": ""
        },
        "overlay_mask": {
            "label": "Overlay mask",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "mask_overlay_blur": {
            "label": "Mask overlay blur",
            "type": "slider",
            "minimum": 0,
            "maximum": 64,
            "step": 1,
            "value": 4,
        },
        "fill": {
            "label": "Mask fill",
            "type": "radio",
            "type_param": "index",
            "choices": ['fill', 'original', 'latent noise', 'latent nothing'],
            "value": 'original',
            "info": ""
        },
        "full_res_mask": {
            "label": "Full res mask",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "full_res_mask_padding": {
            "label": "Full res mask padding",
            "type": "slider",
            "minimum": 0,
            "maximum": 512,
            "step": 1,
            "value": 4,
        },
        "reroll_blank_frames": {
            "label": "Reroll blank frames",
            "type": "radio",
            "choices": ['reroll', 'interrupt', 'ignore'],
            "value": "ignore",
            "info": ""
        },
        "reroll_patience": {
            "label": "Reroll patience",
            "type": "number",
            "precision": None,
            "value": 10,
            "info": ""
        },
        "motion_preview_mode": {
            "label": "Motion preview mode (dry run).",
            "type": "checkbox",
            "value": False,
            "info": "Preview motion only. Uses a static picture for init, and draw motion reference rectangle."
        },
    }


def LoopArgs():
    return {
        "use_looper": {
            "label": "Enable guided images mode",
            "type": "checkbox",
            "value": False,
        },
        "init_images": {
            "label": "Images to use for keyframe guidance",
            "type": "textbox",
            "lines": 9,
            "value": get_guided_imgs_default_json(),
        },
        "image_strength_schedule": {
            "label": "Image strength schedule",
            "type": "textbox",
            "value": "0:(0.85)",
        },
        "image_keyframe_strength_schedule": {
            "label": "Image strength schedule",
            "type": "textbox",
            "value": "0:(0.20)",
        },
        "blendFactorMax": {
            "label": "Blend factor max",
            "type": "textbox",
            "value": "0:(0.35)",
        },
        "blendFactorSlope": {
            "label": "Blend factor slope",
            "type": "textbox",
            "value": "0:(0.25)",
        },
        "tweening_frames_schedule": {
            "label": "Tweening frames schedule",
            "type": "textbox",
            "value": "0:(20)",
        },
        "color_correction_factor": {
            "label": "Color correction factor",
            "type": "textbox",
            "value": "0:(0.075)",
        }
    }


def ParseqArgs():
    return {
        "parseq_manifest": {
            "label": "Parseq manifest (JSON or URL)",
            "type": "textbox",
            "lines": 4,
            "value": None,
        },
        "parseq_use_deltas": {
            "label": "Use delta values for movement parameters (recommended)",
            "type": "checkbox",
            "value": True,
            "info": "Recommended. If you uncheck this, Parseq keyframe values as are treated as relative movement values instead of absolute."
        },
        "parseq_non_schedule_overrides": {
            "label": "Use FPS, max_frames and cadence from the Parseq manifest, if present (recommended)",
            "type": "checkbox",
            "value": True,
            "info": "Recommended. If you uncheck this, the FPS, max_frames and cadence in the Parseq doc are ignored, and the values in the A1111 UI are used instead."
        },
    }


def FreeUArgs():
    return {
        "freeu_enabled": {
            "label": "Enabled",
            "type": "checkbox",
            "value": False,
            "info": "Enable FreeU"
        },
        "freeu_b1": {
            "label": "Backbone stage 1",
            "type": "textbox",
            "value": "0:(1.3)",
            "info": "backbone factor of the first stage block of decoder",
        },
        "freeu_b2": {
            "label": "Backbone stage 2",
            "type": "textbox",
            "value": "0:(1.4)",
            "info": "backbone factor of the second stage block of decoder",
        },
        "freeu_s1": {
            "label": "Skip stage 1",
            "type": "textbox",
            "value": "0:(0.9)",
            "info": "skip factor of the first stage block of decoder",
        },
        "freeu_s2": {
            "label": "Skip stage 2",
            "type": "textbox",
            "value": "0:(0.2)",
            "info": "skip factor of the second stage block of decoder",
        },
    }


def KohyaHRFixArgs():
    return {
        "kohya_hrfix_enabled": {
            "label": "Enabled",
            "type": "checkbox",
            "value": False,
            "info": "Enable Kohya HRFix"
        },
        "kohya_hrfix_block_number": {
            "label": "Block Number (1-32)",
            "type": "textbox",
            "value": "0:(1)",
        },
        "kohya_hrfix_downscale_factor": {
            "label": "Downscale Factor (0.1-9.0)",
            "type": "textbox",
            "value": "0:(2.0)",
        },
        "kohya_hrfix_start_percent": {
            "label": "Start Percent (0.0-1.0)",
            "type": "textbox",
            "value": "0:(0.0)",
        },
        "kohya_hrfix_end_percent": {
            "label": "End Percent (0.0-1.0)",
            "type": "textbox",
            "value": "0:(0.35)",
        },
        "kohya_hrfix_downscale_after_skip": {
            "label": "Downscale After Skip",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "kohya_hrfix_downscale_method": {
            "label": "Downscale Method",
            "type": "radio",
            "choices": ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"],
            "value": "bicubic",
            "info": ""
        },
        "kohya_hrfix_upscale_method": {
            "label": "Upscale Method",
            "type": "radio",
            "choices": ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"],
            "value": "bicubic",
            "info": ""
        }
    }


def WanArgs():
    """Wan 2.1 video generation arguments - Updated to integrate with Deforum schedules"""
    return {
        "wan_t2v_model": {
            "label": "Primary Model",
            "type": "dropdown",
            "choices": ["Auto-Detect", "1.3B VACE", "14B VACE", "1.3B T2V (Legacy)", "14B T2V (Legacy)", "Custom Path"],
            "value": "1.3B VACE",
            "info": "Primary Wan model. VACE 1.3B: 480p, 8GB VRAM. VACE 14B: 480p+720p, 16GB+ VRAM. VACE models handle both T2V and I2V."
        },
        "wan_i2v_model": {
            "label": "I2V Model (Legacy)", 
            "type": "dropdown",
            "choices": ["Auto-Detect", "Use Primary Model", "Use T2V Model (No Continuity)", "14B I2V 720P (Legacy)", "14B I2V 480P (Legacy)", "Custom Path"],
            "value": "Use Primary Model",
            "info": "I2V chaining mode. 'Use Primary Model' uses VACE for seamless transitions (recommended). 'Use T2V Model' = independent clips."
        },
        "wan_auto_download": {
            "label": "Auto-Download Models",
            "type": "checkbox",
            "value": True,
            "info": "Automatically download missing models from HuggingFace (recommended for first-time setup)"
        },
        "wan_preferred_size": {
            "label": "Preferred Model Size",
            "type": "dropdown",
            "choices": ["1.3B VACE (Recommended)", "14B VACE (High Quality)", "Legacy Models"],
            "value": "1.3B VACE (Recommended)",
            "info": "VACE 1.3B: 480p, 8GB VRAM, fast. VACE 14B: 480p+720p, 16GB+ VRAM, better quality."
        },
        "wan_model_path": {
            "label": "Custom Model Path",
            "type": "textbox", 
            "value": "models/wan",
            "info": "Custom path to Wan model (used when 'Custom Path' is selected)"
        },
        "wan_resolution": {
            "label": "Wan Resolution",
            "type": "dropdown",
            "choices": ["864x480 (Landscape)", "480x480 (Landscape)", "512x512 (Landscape)", "480x864 (Portrait)", "1280x720 (Landscape)", "720x1280 (Portrait)"],
            "value": "864x480 (Landscape)",  # Explicit default with label
            "info": "Resolution for Wan video generation. 480p for VACE 1.3B, 720p for VACE 14B. Will warn if mismatched."
        },
        "wan_seed": {
            "label": "Wan Seed",
            "type": "number",
            "precision": 0,
            "value": -1,
            "info": "Seed for Wan generation. -1 for random, 0+ for fixed seed"
        },
        "wan_inference_steps": {
            "label": "Inference Steps",
            "type": "slider",
            "minimum": 5,
            "maximum": 100,
            "step": 1,
            "value": 20,
            "info": "Number of inference steps for Wan generation. Lower values (5-15) for quick testing, higher values (30-50) for quality"
        },
        "wan_strength_override": {
            "label": "Strength Override",
            "type": "checkbox",
            "value": True,
            "info": "Override Deforum strength schedule with fixed value for maximum continuity (recommended for I2V chaining)"
        },
        "wan_fixed_strength": {
            "label": "Fixed Strength",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.05,
            "value": 1.0,
            "info": "Fixed strength value for I2V chaining (1.0 = maximum continuity, 0.0 = maximum creativity)"
        },
        "wan_guidance_override": {
            "label": "Guidance Scale Override",
            "type": "checkbox",
            "value": True,
            "info": "Override Deforum CFG scale schedule with fixed value (recommended for consistent Wan generation)"
        },
        "wan_guidance_scale": {
            "label": "Fixed Guidance Scale",
            "type": "slider",
            "minimum": 1.0,
            "maximum": 20.0,
            "step": 0.5,
            "value": 7.5,
            "info": "Fixed guidance scale for prompt adherence (only used when Guidance Scale Override is enabled)"
        },
        "wan_frame_overlap": {
            "label": "Frame Overlap",
            "type": "slider",
            "minimum": 0,
            "maximum": 10,
            "step": 1,
            "value": 2,
            "info": "Number of overlapping frames between clips for smoother transitions"
        },
        "wan_motion_strength": {
            "label": "Motion Strength",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 2.0,
            "step": 0.1,
            "value": 1.0,
            "info": "Strength of motion in generated videos"
        },
        "wan_enable_interpolation": {
            "label": "Enable Interpolation",
            "type": "checkbox",
            "value": True,
            "info": "Enable frame interpolation between clips"
        },
        "wan_interpolation_strength": {
            "label": "Interpolation Strength",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.1,
            "value": 0.5,
            "info": "Strength of interpolation between consecutive clips"
        },
        "wan_flash_attention_mode": {
            "label": "Flash Attention Mode",
            "type": "dropdown",
            "choices": ["Auto (Recommended)", "Force Flash Attention", "Force PyTorch Fallback"],
            "value": "Auto (Recommended)",
            "info": "Flash Attention mode: Auto tries Flash Attention then falls back to PyTorch. Force options override detection."
        },
        
        # Prompt Enhancement with QwenPromptExpander
        "wan_qwen_model": {
            "label": "Qwen Model",
            "type": "dropdown",
            "choices": ["QwenVL2.5_7B", "QwenVL2.5_3B", "Qwen2.5_14B", "Qwen2.5_7B", "Qwen2.5_3B", "Auto-Select"],
            "value": "Auto-Select",
            "info": "Qwen model for prompt enhancement. VL models support image+text, text-only models are faster. Auto-Select chooses based on available VRAM."
        },
        "wan_qwen_auto_download": {
            "label": "Auto-Download Qwen Models",
            "type": "checkbox",
            "value": True,
            "info": "Automatically download Qwen models if not found locally (recommended for first-time setup)"
        },
        "wan_qwen_language": {
            "label": "Enhanced Prompt Language",
            "type": "dropdown",
            "choices": ["English", "Chinese"],
            "value": "English",
            "info": "Language for enhanced prompts. English works best with most models, Chinese for specialized use cases."
        },
        "wan_enhanced_prompts": {
            "label": "Enhanced Prompts (Editable)",
            "type": "textbox",
            "value": "",
            "lines": 10,
            "info": "Enhanced prompts generated by QwenPromptExpander. You can manually edit these before generation."
        },
        "wan_movement_description": {
            "label": "Movement Description (Auto-Generated)",
            "type": "textbox",
            "value": "",
            "lines": 3,
            "info": "Auto-generated movement description from Deforum schedules (translation, rotation, zoom). Appended to enhanced prompts."
        },
        "wan_movement_sensitivity": {
            "label": "Movement Sensitivity",
            "type": "slider",
            "minimum": 0.1,
            "maximum": 2.0,
            "step": 0.1,
            "value": 1.0,
            "info": "Sensitivity for movement detection. Higher values detect smaller movements, lower values only detect significant motion."
        },
        
        # Advanced Override Settings (moved wan_motion_strength here)
        "wan_motion_strength_override": {
            "label": "Motion Strength Override",
            "type": "checkbox",
            "value": False,
            "info": "Override dynamic motion strength calculation with fixed value"
        },
        "wan_motion_strength": {
            "label": "Fixed Motion Strength",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 2.0,
            "step": 0.1,
            "value": 1.0,
            "info": "Fixed motion strength value (only used when Motion Strength Override is enabled). Dynamic calculation from schedules is recommended."
        }
    }


def DeforumOutputArgs():
    return {
        "skip_video_creation": {
            "label": "Skip video creation",
            "type": "checkbox",
            "value": False,
            "info": "If enabled, only images will be saved"
        },
        "fps": {
            "label": "FPS",
            "type": "slider",
            "minimum": 1,
            "maximum": 240,
            "step": 1,
            "value": 60,
        },
        "make_gif": {
            "label": "Make GIF",
            "type": "checkbox",
            "value": False,
            "info": "make GIF in addition to the video/s"
        },
        "delete_imgs": {
            "label": "Delete Imgs",
            "type": "checkbox",
            "value": False,
            "info": "auto-delete imgs when video is ready. Will break Resume from timestring!"
        },
        "delete_input_frames": {
            "label": "Delete All Inputframes",
            "type": "checkbox",
            "value": False,
            "info": "auto-delete inputframes (incl CN ones) when video is ready"
        },
        "image_path": {
            "label": "Image path",
            "type": "textbox",
            "value": "C:/SD/20241111111111_%09d.png",
        },
        "add_soundtrack": {
            "label": "Add soundtrack",
            "type": "radio",
            "choices": ['None', 'File', 'Init Video'],
            "value": "File",
            "info": "add audio to video from file/url or init video"
        },
        "soundtrack_path": {
            "label": "Soundtrack path",
            "type": "textbox",
            "value": "https://ia801303.us.archive.org/26/items/amen-breaks/cw_amen13_173.mp3",
            "info": "abs. path or url to audio file"
        },
        "r_upscale_video": {
            "label": "Upscale",
            "type": "checkbox",
            "value": False,
            "info": "upscale output imgs when run is finished"
        },
        "r_upscale_factor": {
            "label": "Upscale factor",
            "type": "dropdown",
            "choices": ['x2', 'x3', 'x4'],
            "value": "x2",
        },
        "r_upscale_model": {
            "label": "Upscale model",
            "type": "dropdown",
            "choices": ['realesr-animevideov3', 'realesrgan-x4plus', 'realesrgan-x4plus-anime'],
            "value": 'realesr-animevideov3',
        },
        "r_upscale_keep_imgs": {
            "label": "Keep Imgs",
            "type": "checkbox",
            "value": True,
            "info": "don't delete upscaled imgs",
        },
        "store_frames_in_ram": {
            "label": "Store frames in ram",
            "type": "checkbox",
            "value": False,
            "info": "auto-delete imgs when video is ready",
            "visible": False
        },
        "frame_interpolation_engine": {
            "label": "Engine",
            "type": "radio",
            "choices": ['None', 'RIFE v4.6', 'FILM'],
            "value": "None",
            "info": "select the frame interpolation engine. hover on the options for more info"
        },
        "frame_interpolation_x_amount": {
            "label": "Interp X",
            "type": "slider",
            "minimum": 2,
            "maximum": 10,
            "step": 1,
            "value": 2,
        },
        "frame_interpolation_slow_mo_enabled": {
            "label": "Slow-Mo",
            "type": "checkbox",
            "value": False,
            "visible": False,
            "info": "Slow-Mo the interpolated video, audio will not be used if enabled",
        },
        "frame_interpolation_slow_mo_amount": {
            "label": "Slow-Mo X",
            "type": "slider",
            "minimum": 2,
            "maximum": 10,
            "step": 1,
            "value": 2,
        },
        "frame_interpolation_keep_imgs": {
            "label": "Keep Imgs",
            "type": "checkbox",
            "value": False,
            "info": "Keep interpolated images on disk",
            "visible": False
        },
        "frame_interpolation_use_upscaled": {
            "label": "Use Upscaled",
            "type": "checkbox",
            "value": False,
            "info": "Interpolate upscaled images, if available",
            "visible": False
        },
    }



def FramePackF1Args():
    """Arguments specific to FramePack F1 mode"""
    lora_dir = sh.cmd_opts.lora_dir # 動的なパスを使用
    try:
        lora_choices = [f for f in os.listdir(lora_dir) if f.endswith((".safetensors", ".pt"))]
    except Exception:
        lora_choices = []

    return {
        "f1_image_strength": {
            "label": "Image Strength (F1 Mode)",
            "type": "slider",
            "minimum": 0.99,
            "maximum": 1.01,
            "step": 0.0001,
            "value": 1.0,
            "info": "Influence of the initial image. Higher values stick closer to the start image. (1.0 = 100%)",
        },
        "f1_generation_latent_size": {
            "label": "Generation Latent Size (F1 Mode)",
            "type": "slider",
            "minimum": 1,
            "maximum": 12,
            "step": 1,
            "value": 9,
            "info": "Frames to generate to connect to the initial image. (Recommended: 6-9)",
        },
        "f1_trim_start_latent_size": {
            "label": "Trim Start Frames (F1 Mode)",
            "type": "slider",
            "minimum": 0,
            "maximum": 5,
            "step": 1,
            "value": 0,
            "info": "Frames to trim from the beginning of the video (if noise is present).",
        },
        "lora_path_1": {
            "label": "LoRA Slot 1",
            "type": "dropdown",
            "choices": ["None"] + lora_choices,
            "value": "None",
            "info": "Select first LoRA file or leave as None",
        },
        "lora_weight_1": {
            "label": "Weight 1",
            "type": "number",
            "precision": 2,
            "value": 1.0,
            "info": "Weight for first LoRA",
        },
        "lora_path_2": {
            "label": "LoRA Slot 2",
            "type": "dropdown",
            "choices": ["None"] + lora_choices,
            "value": "None",
            "info": "Select second LoRA file or leave as None",
        },
        "lora_weight_2": {
            "label": "Weight 2",
            "type": "number",
            "precision": 2,
            "value": 1.0,
            "info": "Weight for second LoRA",
        },
        "lora_path_3": {
            "label": "LoRA Slot 3",
            "type": "dropdown",
            "choices": ["None"] + lora_choices,
            "value": "None",
            "info": "Select third LoRA file or leave as None",
        },
        "lora_weight_3": {
            "label": "Weight 3",
            "type": "number",
            "precision": 2,
            "value": 1.0,
            "info": "Weight for third LoRA",
        },
    }

def get_component_names():
    # Re-enable Wan components (UI level, imports still isolated)
    return ['override_settings_with_file', 'custom_settings_file', *DeforumAnimArgs().keys(), 'animation_prompts',
            'animation_prompts_positive', 'animation_prompts_negative',
            *DeforumArgs().keys(), *DeforumOutputArgs().keys(), *ParseqArgs().keys(), *LoopArgs().keys(),
            *controlnet_component_names(), *FreeUArgs().keys(), *KohyaHRFixArgs().keys(), *WanArgs().keys(),
            *FramePackF1Args().keys()]


def get_settings_component_names():
    return [name for name in get_component_names()]


def pack_args(args_dict, keys_function):
    return {name: args_dict[name] for name in keys_function()}


def process_args(args_dict_main, run_id):
    from .settings import load_args
    override_settings_with_file = args_dict_main['override_settings_with_file']
    custom_settings_file = args_dict_main['custom_settings_file']
    p = args_dict_main['p']

    root = SimpleNamespace(**RootArgs())
    args = SimpleNamespace(**{name: args_dict_main[name] for name in DeforumArgs()})
    anim_args = SimpleNamespace(**{name: args_dict_main[name] for name in DeforumAnimArgs()})
    video_args = SimpleNamespace(**{name: args_dict_main[name] for name in DeforumOutputArgs()})
    parseq_args = SimpleNamespace(**{name: args_dict_main[name] for name in ParseqArgs()})
    loop_args = SimpleNamespace(**{name: args_dict_main[name] for name in LoopArgs()})
    freeu_args = SimpleNamespace(**{name: args_dict_main[name] for name in FreeUArgs()})
    kohya_hrfix_args = SimpleNamespace(**{name: args_dict_main[name] for name in KohyaHRFixArgs()})
    wan_args = SimpleNamespace(**{name: args_dict_main[name] for name in WanArgs()})
    framepack_f1_args = SimpleNamespace(**{name: args_dict_main[name] for name in FramePackF1Args()})
    controlnet_args = SimpleNamespace(**{name: args_dict_main[name] for name in controlnet_component_names()})

    # Build LoRA path and scale lists from individual slots
    lora_paths = []
    lora_scales = []
    lora_base = sh.cmd_opts.lora_dir # 動的なパスを使用
    for i in range(1, 4):
        path = getattr(framepack_f1_args, f"lora_path_{i}", "None")
        if path and path != "None":
            full_path = path if os.path.isabs(path) else os.path.join(lora_base, path)
            lora_paths.append(full_path)
            scale = getattr(framepack_f1_args, f"lora_weight_{i}", 1.0)
            try:
                lora_scales.append(float(scale))
            except Exception:
                lora_scales.append(1.0)

    framepack_f1_args.lora_paths = lora_paths
    framepack_f1_args.lora_scales = lora_scales

    root.animation_prompts = json.loads(args_dict_main['animation_prompts'])

    args_loaded_ok = True
    if override_settings_with_file:
        args_loaded_ok = load_args(args_dict_main, args, anim_args, parseq_args, loop_args, controlnet_args, freeu_args,
                                   kohya_hrfix_args, video_args, custom_settings_file, root, run_id)

    positive_prompts = args_dict_main['animation_prompts_positive']
    negative_prompts = args_dict_main['animation_prompts_negative']
    negative_prompts = negative_prompts.replace('--neg',
                                                '')  # remove --neg from negative_prompts if received by mistake
    root.prompt_keyframes = [key for key in root.animation_prompts.keys()]
    root.animation_prompts = {key: f"{positive_prompts} {val} {'' if '--neg' in val else '--neg'} {negative_prompts}"
                              for key, val in root.animation_prompts.items()}

    if args.seed == -1:
        root.raw_seed = -1
    args.seed = get_fixed_seed(args.seed)
    if root.raw_seed != -1:
        root.raw_seed = args.seed
    root.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))
    args.prompts = json.loads(args_dict_main['animation_prompts'])
    args.positive_prompts = args_dict_main['animation_prompts_positive']
    args.negative_prompts = args_dict_main['animation_prompts_negative']

    if not args.use_init and not anim_args.hybrid_use_init_image:
        args.init_image = None
        args.init_image_box = None

    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True

    additional_substitutions = SimpleNamespace(date=time.strftime('%Y%m%d'), time=time.strftime('%H%M%S'))
    current_arg_list = [args, anim_args, video_args, parseq_args, root, additional_substitutions]
    full_base_folder_path = os.path.join(os.getcwd(), p.outpath_samples)
    root.raw_batch_name = args.batch_name
    args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)
    args.outdir = os.path.join(p.outpath_samples, str(args.batch_name))
    args.outdir = os.path.join(os.getcwd(), args.outdir)
    args.outdir = os.path.realpath(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    default_img = Image.open(os.path.join(pathlib.Path(__file__).parent.absolute(), '114763196.jpg'))
    assert default_img is not None
    default_img = default_img.resize((args.W, args.H))
    root.default_img = default_img

    return args_loaded_ok, root, args, anim_args, video_args, parseq_args, loop_args, controlnet_args, freeu_args, kohya_hrfix_args, wan_args, framepack_f1_args
