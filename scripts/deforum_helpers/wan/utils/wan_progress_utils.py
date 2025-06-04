"""
Wan Progress Utilities - Styled progress indicators matching experimental render core
"""

import sys
from tqdm import tqdm
from typing import Optional, Any

# Import the same color constants as experimental render core
# Assuming the relative import path is correct for your project structure.
# If not, you might need to adjust it (e.g., from deforum_helpers.rendering.util.log_utils import ...)
try:
    from ...rendering.util.log_utils import (
        HEX_BLUE, HEX_GREEN, HEX_ORANGE, HEX_RED, HEX_PURPLE, HEX_YELLOW,
        BLUE, GREEN, ORANGE, RED, PURPLE, YELLOW, RESET_COLOR, BOLD
    )
except ImportError:
    # Fallback or placeholder if the import fails, to allow the script to be parsed
    # In a real environment, this path should be resolvable.
    print("Warning: Could not import color constants from ...rendering.util.log_utils. Using placeholders.")
    HEX_BLUE, HEX_GREEN, HEX_ORANGE, HEX_RED, HEX_PURPLE, HEX_YELLOW = ["#0000FF"] * 6
    BLUE, GREEN, ORANGE, RED, PURPLE, YELLOW, RESET_COLOR, BOLD = [""] * 8


# Import shared WebUI progress handling
try:
    import modules.shared as shared
    WEBUI_AVAILABLE = True
except ImportError:
    WEBUI_AVAILABLE = False
    shared = None # type: ignore


class WanProgressBar:
    """Styled progress bar for Wan operations using experimental render core colors"""
    
    # Progress bar formats matching experimental render core
    NO_ETA_RBAR = "| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"
    NO_ETA_BAR_FORMAT = "{l_bar}{bar}" + f"{NO_ETA_RBAR}"
    DEFAULT_BAR_FORMAT = "{l_bar}{bar}{r_bar}"
    
    def __init__(self, total: int, description: str, color: str = HEX_BLUE, 
                 unit: str = "step", position: int = 0, 
                 bar_format: Optional[str] = None):
        """
        Create a styled progress bar for Wan operations
        
        Args:
            total: Total number of items
            description: Progress bar description
            color: Hex color (use HEX_* constants)
            unit: Unit name for progress
            position: Position for multi-bar displays
            bar_format: Custom bar format (defaults to NO_ETA_BAR_FORMAT)
        """
        self.total = total
        self.description = description
        self.color = color
        self.unit = unit
        self.position = position
        self.bar_format = bar_format or self.NO_ETA_BAR_FORMAT
        self.pbar: Optional[tqdm] = None
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def start(self):
        """Start the progress bar"""
        if self.pbar is not None:
            return
            
        # Use WebUI's progress output if available
        file_output = shared.progress_print_out if WEBUI_AVAILABLE and shared and hasattr(shared, 'progress_print_out') else sys.stdout
        
        # Check if console progress bars are disabled
        disable_console = (WEBUI_AVAILABLE and 
                          shared and
                          hasattr(shared, 'cmd_opts') and 
                          getattr(shared.cmd_opts, 'disable_console_progressbars', False))
        
        self.pbar = tqdm(
            total=self.total,
            desc=self.description,
            unit=self.unit,
            position=self.position,
            dynamic_ncols=True,
            file=file_output,
            bar_format=self.bar_format,
            disable=disable_console,
            colour=self.color
        )
        
    def update(self, n: int = 1, postfix: Optional[str] = None):
        """Update progress bar"""
        if self.pbar:
            if postfix:
                self.pbar.set_postfix_str(postfix) # Use tqdm's set_postfix_str for simple string postfix
            self.pbar.update(n)
            
    def set_description(self, desc: str):
        """Update description"""
        if self.pbar:
            self.pbar.set_description(desc)
            
    def set_postfix(self, **kwargs):
        """Set postfix with key-value pairs. Uses tqdm's set_postfix."""
        if self.pbar:
            self.pbar.set_postfix(**kwargs)

    def set_postfix_str(self, s: str = "", refresh: bool = True):
        """Set postfix with a string. Uses tqdm's set_postfix_str."""
        if self.pbar:
            self.pbar.set_postfix_str(s, refresh=refresh)
            
    def close(self):
        """Close the progress bar"""
        if self.pbar:
            self.pbar.close()
            self.pbar = None


def print_wan_info(message: str, color: str = BLUE):
    """Print Wan info message with styling"""
    print(f"{color}{BOLD}Wan: {RESET_COLOR}{message}")

    
def print_wan_success(message: str):
    """Print Wan success message"""
    print_wan_info(f"✅ {message}", GREEN)

    
def print_wan_warning(message: str):
    """Print Wan warning message"""
    print_wan_info(f"⚠️ {message}", ORANGE)

    
def print_wan_error(message: str):
    """Print Wan error message"""
    print_wan_info(f"❌ {message}", RED)

    
def print_wan_progress(message: str):
    """Print Wan progress message"""
    print_wan_info(f"🎬 {message}", PURPLE)


def create_wan_model_loader_progress(model_name: str) -> WanProgressBar:
    """Create progress bar for model loading"""
    return WanProgressBar(
        total=100,  # Percentage-based
        description=f"Loading {model_name}",
        color=HEX_BLUE,
        unit="%"
    )


def create_wan_clip_progress(total_clips: int) -> WanProgressBar:
    """Create progress bar for clip generation"""
    return WanProgressBar(
        total=total_clips,
        description="Generating Clips",
        color=HEX_PURPLE,
        unit="clip",
        position=0 # Main progress for clips
    )


def create_wan_frame_progress(total_frames: int, clip_idx: int) -> WanProgressBar:
    """Create progress bar for frame generation within a clip"""
    # Position 1 to appear below the clip progress bar
    return WanProgressBar(
        total=total_frames,
        description=f"Clip {clip_idx + 1} Frames",
        color=HEX_GREEN,
        unit="frame",
        position=1 
    )


def create_wan_inference_progress(total_steps: int) -> WanProgressBar:
    """Create progress bar for inference steps"""
    # Position 2 to appear below frame progress if used simultaneously
    return WanProgressBar(
        total=total_steps,
        description="Inference Steps",
        color=HEX_ORANGE,
        unit="step",
        position=2
    )


def create_wan_video_processing_progress(total_frames: int) -> WanProgressBar:
    """Create progress bar for video post-processing"""
    return WanProgressBar(
        total=total_frames,
        description="Processing Video",
        color=HEX_RED,
        unit="frame"
    )


# Context managers for common Wan operations
class WanModelLoadingContext:
    """Context manager for model loading with progress"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pbar: Optional[WanProgressBar] = None # Type hint for clarity
        
    def __enter__(self) -> Optional[WanProgressBar]: # Return type hint
        print_wan_progress(f"Loading model: {self.model_name}")
        self.pbar = create_wan_model_loader_progress(self.model_name)
        self.pbar.start()
        return self.pbar
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            if exc_type is None:
                self.pbar.update(100)  # Complete to 100%
                # Ensure description reflects completion if necessary
                self.pbar.set_description(f"Loaded {self.model_name}")
                print_wan_success(f"Model loaded: {self.model_name}")
            else:
                self.pbar.set_description(f"Failed {self.model_name}") # Update description on failure
                print_wan_error(f"Failed to load model: {self.model_name} - {exc_val}")
            self.pbar.close()


class WanGenerationContext:
    """Context manager for video generation with progress"""
    
    def __init__(self, total_clips: int):
        self.total_clips = total_clips
        self.clip_pbar: Optional[WanProgressBar] = None
        # Initialize current_clip_label if it's intended to be part of this class
        # self.current_clip_label: Optional[Any] = None # Example, if it's a UI element
        
    def __enter__(self) -> 'WanGenerationContext': # Return self for use with 'as'
        print_wan_progress(f"Starting generation of {self.total_clips} clips")
        self.clip_pbar = create_wan_clip_progress(self.total_clips)
        self.clip_pbar.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.clip_pbar:
            if exc_type is None:
                # Ensure the progress bar reaches 100% if all clips were processed
                # This might require knowing if all clips were actually updated.
                # If update_clip is called for each clip, it should naturally complete.
                print_wan_success(f"Generated {self.total_clips} clips successfully")
            else:
                print_wan_error(f"Generation failed after some clips: {exc_val}")
            self.clip_pbar.close()
            
    # Corrected method signature and internal variable name
    def update_clip(self, clip_idx: int, prompt_preview: str):
        """Update clip progress and description."""
        # Debug log using the corrected variable name 'prompt_preview'
        print(f"DEBUG: update_clip called with clip_idx={clip_idx}, prompt_preview='{prompt_preview}'")

        if self.clip_pbar:
            self.clip_pbar.update(1) # Increment progress by 1 clip
            
            try:
                # Debug log for formatting postfix
                print(f"DEBUG: Formatting postfix with prompt_preview='{prompt_preview}'")
                
                # Truncate prompt_preview if it's too long for display
                max_prompt_len = 30 # Adjustable
                display_prompt = (prompt_preview[:max_prompt_len] + '...') if len(prompt_preview) > max_prompt_len else prompt_preview
                
                postfix_str = f"Clip {clip_idx + 1}/{self.total_clips} ({display_prompt})"
                
                # Debug log for generated postfix string
                print(f"DEBUG: Generated postfix_str='{postfix_str}'")

                # Use the new set_postfix_str method of WanProgressBar
                self.clip_pbar.set_postfix_str(postfix_str, refresh=True)
            except Exception as e:
                print(f"ERROR in update_clip while setting postfix: {e}")
                import traceback
                traceback.print_exc()

            # The 'current_clip_label' attribute is not initialized in __init__.
            # If this is a UI element, it should be passed in or handled more robustly.
            # For now, adding a check to prevent AttributeError if it's not set.
            if hasattr(self, 'current_clip_label') and self.current_clip_label is not None:
                try:
                    # Assuming current_clip_label has a 'value' attribute if it's a UI element
                    self.current_clip_label.value = f"Processing Clip: {clip_idx + 1}/{self.total_clips}"
                except Exception as e:
                    print(f"ERROR in update_clip while setting current_clip_label: {e}")


# Utility functions for common print replacements
def replace_wan_prints_with_styled(text: str) -> str:
    """Replace common Wan print patterns with styled versions"""
    # Ensure RESET_COLOR is appended where colors are used without explicit reset
    replacements = {
        "🎬 Wan": f"{PURPLE}{BOLD}🎬 Wan{RESET_COLOR}",
        "✅": f"{GREEN}✅{RESET_COLOR}", # Success
        "❌": f"{RED}❌{RESET_COLOR}",   # Error
        "⚠️": f"{ORANGE}⚠️{RESET_COLOR}",# Warning
        "🔧": f"{BLUE}🔧{RESET_COLOR}",  # Tool/Config
        "🔍": f"{YELLOW}🔍{RESET_COLOR}",# Info/Search
        "📁": f"{GREEN}📁{RESET_COLOR}", # File/Directory (often success related)
        "📝": f"{BLUE}📝{RESET_COLOR}",  # Notes/Params
        "🎯": f"{PURPLE}🎯{RESET_COLOR}",# Target/Goal
    }
    
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    import time

    print_wan_info("Starting Wan Utility Test")

    # Test WanModelLoadingContext
    with WanModelLoadingContext(model_name="TestModel-1.0") as pbar:
        if pbar: # Check if pbar was created
            for i in range(100):
                time.sleep(0.01)
                pbar.update(1)
                if i % 10 == 0:
                    pbar.set_postfix_str(f"Loading file {i+1}/100")
    
    print("-" * 20)
    
    # Test WanGenerationContext
    total_clips_to_generate = 5
    prompts = [
        "A beautiful sunrise over mountains",
        "A futuristic cityscape at night with flying cars",
        "A serene forest with a hidden waterfall",
        "A bustling medieval marketplace with various characters",
        "An abstract animation of swirling colors and shapes"
    ]

    # Example of how current_clip_label might be handled if it were a simple mock object
    class MockLabel:
        def __init__(self):
            self.value = ""
        def __str__(self):
            return f"MockLabel: {self.value}"

    mock_ui_label = MockLabel()

    with WanGenerationContext(total_clips=total_clips_to_generate) as gen_ctx:
        # If current_clip_label is used, it should be set on the context object
        # gen_ctx.current_clip_label = mock_ui_label # Example

        for i in range(total_clips_to_generate):
            current_prompt = prompts[i % len(prompts)]
            print_wan_progress(f"Preparing to generate clip {i+1} with prompt: {current_prompt[:30]}...")
            gen_ctx.update_clip(clip_idx=i, prompt_preview=current_prompt)
            
            # Simulate frame generation progress for this clip
            total_frames_in_clip = 20
            # Create a frame progress bar manually if needed, or integrate into context
            with create_wan_frame_progress(total_frames_in_clip, clip_idx=i) as frame_pbar:
                 for frame_num in range(total_frames_in_clip):
                    time.sleep(0.02) # Simulate work for each frame
                    frame_pbar.update(1)
                    if frame_num % 5 == 0:
                        frame_pbar.set_postfix_str(f"Detail {frame_num}")
            
            # print(str(mock_ui_label)) # Print the mock label's value after update
            print_wan_success(f"Completed processing for clip {i+1}")
            time.sleep(0.1) # Simulate time between clips

    print("-" * 20)
    print_wan_info("Wan Utility Test Finished")

    # Test styled print replacements
    test_string = "🎬 Wan is starting. ✅ Process A complete. ❌ Error in Process B. ⚠️ Warning: Low disk space. 🔧 Settings updated."
    styled_string = replace_wan_prints_with_styled(test_string)
    print("\nTesting styled prints:")
    print(f"Original: {test_string}")
    print(f"Styled:   {styled_string}")

