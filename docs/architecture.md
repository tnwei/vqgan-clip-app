# Architecture

## App structure

There are three major components in this repo:

+ VQGAN-CLIP app
    + `app.py` houses the UI
    + `logic.py` stores underlying logic
    + `vqgan_utils.py` stores utility functions used by `logic.py`
+ CLIP guided diffusion app
    + `diffusion_app.py` houses the UI
    + `diffusion_logic.py` stores underlying logic
+ Gallery viewer
    + `gallery.py` houses the UI
    + `templates/index.html` houses the HTML page template

To date, all three components are independent and do not have shared dependencies. 

The UI for both image generation apps are built using Streamlit ([docs](https://docs.streamlit.io/en/stable/index.html)), which makes it easy to throw together dashboards for ML projects in a short amount of time. The gallery viewer is a simple Flask dashboard. 

## Customizing this repo

Defaults settings for the app upon launch are specified in `defaults.yaml`, which can be adjusted as necessary.

To use customized weights for VQGAN-CLIP, save both the `yaml` config and the `ckpt` weights in `assets/`, and ensure that they have the same filename, just with different postfixes (e.g. `mymodel.ckpt`, `mymodel.yaml`). It will then appear in the app interface for use. Refer to the [VQGAN repo](https://github.com/CompVis/taming-transformers) for training VQGAN on your own dataset.

To modify the image generation logic, instantiate your own model class in `logic.py` / `diffusion_logic.py` and modify `app.py` / `diffusion_app.py` accordingly if changes to UI elements are needed.