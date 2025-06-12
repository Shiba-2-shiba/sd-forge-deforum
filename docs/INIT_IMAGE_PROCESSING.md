# Init Image Processing and Custom Mode Development

This document explains how the **init_image** workflow operates in Deforum and provides
pointers for implementing new animation modes that correctly leverage this logic.

## Updated Init Image Workflow

1. **Input from UI**
   - Users can supply a path/URL in `init_image` or upload an image via
     `init_image_box`. Uploaded images take priority.
2. **Image Loading**
   - `generate.py` calls `load_img()` from `load_images.py` to read the image.
   - `load_img()` internally invokes `load_image()` which returns the uploaded
     image if available. Otherwise the path or URL is processed. Remote images
     are downloaded when internet access is available.
3. **Resize and Alpha Mask**
   - The image is resized to `(args.W, args.H)` using the `Image.LANCZOS`
     resampling filter.
   - When `use_alpha_as_mask` is true, the alpha channel is separated and
     returned as `mask_image` while the original image is converted to RGB.
4. **Passing to the Pipeline**
   - The resized `PIL.Image` is provided to Forge's `img2img` pipeline. This only
     happens for the very first frame when `use_init` is enabled.

```python
# generate.py
init_image, mask_image = load_img(
    image_init0,
    image_init0_box,
    shape=(args.W, args.H),
    use_alpha_as_mask=args.use_alpha_as_mask,
)
p.init_images = [init_image]
processed = processing.process_images(p)
```

## Notes for New Animation Modes

When adding a new mode you should re-use the existing init image logic instead of
re-implementing it.

1. **Define the Mode**
   - In `args.py` add your mode name to the `animation_mode` choices.

   ```python
   "animation_mode": {
       "label": "Animation mode",
       "type": "radio",
       "choices": ['2D', '3D', 'Video Input', 'Interpolation', 'Wan Video', 'FramePack F1', 'YourNewMode'],
       "value": "2D",
   }
   ```
2. **Add Branching Logic**
   - Update `run_deforum.py` so your mode calls a dedicated rendering function.

   ```python
   if anim_args.animation_mode == 'YourNewMode':
       render_your_new_mode(args, anim_args, ...)
   ```
3. **Use `generate()` for Frames**
   - For frame 0 call `generate()` with `args.use_init` enabled and the user
     supplied `init_image` or `init_image_box` present.
   - For subsequent frames place the previous output (after any custom
     processing) into `root.init_sample` before calling `generate()` again.
   - `generate()` detects `root.init_sample` and uses it in place of the user
     provided `init_image`.

Key variables to be aware of:
- `args.use_init`: whether to use the init image at all.
- `args.init_image` / `args.init_image_box`: user supplied images.
- `root.init_sample`: processed image fed to the next frame.
- `prev_img`: numpy array containing the previous frame output.

Following these guidelines ensures your mode benefits from Deforum's robust image
loading and processing pipeline.
