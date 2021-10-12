"""
This script is organized like so:
+ `if __name__ == "__main__" sets up the Streamlit UI elements
+ `generate_image` houses interactions between UI and the CLIP image 
generation models
+ Core model code is abstracted in `logic.py` and imported in `generate_image`
"""
import streamlit as st
from pathlib import Path
import sys
import datetime
import shutil
import json
import os
import base64

sys.path.append("./taming-transformers")

from PIL import Image
from typing import Optional, List
from omegaconf import OmegaConf
import imageio
import numpy as np
from mselogic import MSEVQGANCLIPRun


def generate_image(
    text_input: str = "the first day of the waters",
    vqgan_ckpt: str = "vqgan_imagenet_f16_16384",
    num_steps: int = 300,
    image_x: int = 300,
    image_y: int = 300,
    init_image: Optional[Image.Image] = None,
    image_prompts: List[Image.Image] = [],
    continue_prev_run: bool = False,
    seed: Optional[int] = None,
) -> None:

    ### Init -------------------------------------------------------------------
    run = MSEVQGANCLIPRun(
        text_input=text_input,
        vqgan_ckpt=vqgan_ckpt,
        num_steps=num_steps,
        image_x=image_x,
        image_y=image_y,
        seed=seed,
        init_image=init_image,
        image_prompts=image_prompts,
        continue_prev_run=continue_prev_run,
        use_augs=True,
        noise_fac=0.1,
        use_noise=None,
        mse_withzeros=True,
        mse_decay_rate=50,
        mse_epoches=5,
    )

    ### Load model -------------------------------------------------------------

    if continue_prev_run is True:
        run.load_model(
            prev_model=st.session_state["model"],
            prev_perceptor=st.session_state["perceptor"],
        )
        prev_run_id = st.session_state["run_id"]

    else:
        # Remove the cache first! CUDA out of memory
        if "model" in st.session_state:
            del st.session_state["model"]

        if "perceptor" in st.session_state:
            del st.session_state["perceptor"]

        st.session_state["model"], st.session_state["perceptor"] = run.load_model()
        prev_run_id = None

    # Generate random run ID
    # Used to link runs linked w/ continue_prev_run
    # ref: https://stackoverflow.com/a/42703382/13095028
    # Use URL and filesystem safe version since we're using this as a folder name
    run_id = st.session_state["run_id"] = base64.urlsafe_b64encode(
        os.urandom(6)
    ).decode("ascii")

    run_start_dt = datetime.datetime.now()

    ### Model init -------------------------------------------------------------
    if continue_prev_run is True:
        run.model_init(init_image=st.session_state["prev_im"])
    elif init_image is not None:
        run.model_init(init_image=init_image)
    else:
        run.model_init()

    ### Iterate ----------------------------------------------------------------
    step_counter = 0
    frames = []

    try:
        # Try block catches st.script_runner.StopExecution, no need of a dedicated stop button
        # Reason is st.form is meant to be self-contained either within sidebar, or in main body
        # The way the form is implemented in this app splits the form across both regions
        # This is intended to prevent the model settings from crowding the main body
        # However, touching any button resets the app state, making it impossible to
        # implement a stop button that can still dump output
        # Thankfully there's a built-in stop button :)
        while True:
            # While loop to accomodate running predetermined steps or running indefinitely
            status_text.text(f"Running step {step_counter}")

            _, im = run.iterate()

            if num_steps > 0:  # skip when num_steps = -1
                step_progress_bar.progress((step_counter + 1) / num_steps)
            else:
                step_progress_bar.progress(100)

            # At every step, display and save image
            im_display_slot.image(im, caption="Output image", output_format="PNG")
            st.session_state["prev_im"] = im

            # ref: https://stackoverflow.com/a/33117447/13095028
            # im_byte_arr = io.BytesIO()
            # im.save(im_byte_arr, format="JPEG")
            # frames.append(im_byte_arr.getvalue()) # read()
            frames.append(np.asarray(im))

            step_counter += 1

            if (step_counter == num_steps) and num_steps > 0:
                break

        # Stitch into video using imageio
        writer = imageio.get_writer("temp.mp4", fps=24)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        # Save to output folder if run completed
        runoutputdir = outputdir / (
            run_start_dt.strftime("%Y%m%dT%H%M%S") + "-" + run_id
        )
        runoutputdir.mkdir()

        # Save final image
        im.save(runoutputdir / "output.PNG", format="PNG")

        # Save init image
        if init_image is not None:
            init_image.save(runoutputdir / "init-image.JPEG", format="JPEG")

        # Save image prompts
        for count, image_prompt in enumerate(image_prompts):
            image_prompt.save(
                runoutputdir / f"image-prompt-{count}.JPEG", format="JPEG"
            )

        # Save animation
        shutil.copy("temp.mp4", runoutputdir / "anim.mp4")

        # Save metadata
        with open(runoutputdir / "details.json", "w") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "num_steps": step_counter,
                    "planned_num_steps": num_steps,
                    "text_input": text_input,
                    "init_image": False if init_image is None else True,
                    "image_prompts": False if len(image_prompts) == 0 else True,
                    "continue_prev_run": continue_prev_run,
                    "prev_run_id": prev_run_id,
                    "seed": run.seed,
                    "Xdim": image_x,
                    "ydim": image_y,
                    "vqgan_ckpt": vqgan_ckpt,
                    "start_time": run_start_dt.strftime("%Y%m%dT%H%M%S"),
                    "end_time": datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
                },
                f,
                indent=4,
            )

        status_text.text("Done!")  # End of run

    except st.script_runner.StopException as e:
        # Dump output to dashboard
        print(f"Received Streamlit StopException")
        status_text.text("Execution interruped, dumping outputs ...")
        writer = imageio.get_writer("temp.mp4", fps=24)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        # TODO: Make the following DRY
        # Save to output folder if run completed
        runoutputdir = outputdir / (
            run_start_dt.strftime("%Y%m%dT%H%M%S") + "-" + run_id
        )
        runoutputdir.mkdir()

        # Save final image
        im.save(runoutputdir / "output.PNG", format="PNG")

        # Save init image
        if init_image is not None:
            init_image.save(runoutputdir / "init-image.JPEG", format="JPEG")

        # Save image prompts
        for count, image_prompt in enumerate(image_prompts):
            image_prompt.save(
                runoutputdir / f"image-prompt-{count}.JPEG", format="JPEG"
            )

        # Save animation
        shutil.copy("temp.mp4", runoutputdir / "anim.mp4")

        # Save metadata
        with open(runoutputdir / "details.json", "w") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "num_steps": step_counter,
                    "planned_num_steps": num_steps,
                    "text_input": text_input,
                    "init_image": False if init_image is None else True,
                    "image_prompts": False if len(image_prompts) == 0 else True,
                    "continue_prev_run": continue_prev_run,
                    "prev_run_id": prev_run_id,
                    "seed": run.seed,
                    "Xdim": image_x,
                    "ydim": image_y,
                    "vqgan_ckpt": vqgan_ckpt,
                    "start_time": run_start_dt.strftime("%Y%m%dT%H%M%S"),
                    "end_time": datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
                },
                f,
                indent=4,
            )
        status_text.text("Done!")  # End of run


if __name__ == "__main__":
    defaults = OmegaConf.load("defaults.yaml")
    outputdir = Path("output")
    if not outputdir.exists():
        outputdir.mkdir()

    st.set_page_config(page_title="VQGAN-CLIP playground")
    st.title("VQGAN-CLIP playground")

    # Determine what weights are available in `assets/`
    weights_dir = Path("assets").resolve()
    available_weight_ckpts = list(weights_dir.glob("*.ckpt"))
    available_weight_configs = list(weights_dir.glob("*.yaml"))
    available_weights = [
        i.stem
        for i in available_weight_ckpts
        if i.stem in [j.stem for j in available_weight_configs]
    ]

    # Set vqgan_imagenet_f16_1024 as default if possible
    if "vqgan_imagenet_f16_1024" in available_weights:
        default_weight_index = available_weights.index("vqgan_imagenet_f16_1024")
    else:
        default_weight_index = 0

    # Start of input form
    with st.form("form-inputs"):
        # Only element not in the sidebar, but in the form
        text_input = st.text_input(
            "Text prompt",
            help="VQGAN-CLIP will generate an image that best fits the prompt",
        )
        radio = st.sidebar.radio(
            "Model weights",
            available_weights,
            index=default_weight_index,
            help="Choose which weights to load, trained on different datasets. Make sure the weights and configs are downloaded to `assets/` as per the README!",
        )
        num_steps = st.sidebar.number_input(
            "Num steps",
            value=defaults["num_steps"],
            min_value=-1,
            max_value=None,
            step=1,
            help="Specify -1 to run indefinitely. Use Streamlit's stop button in the top right corner to terminate execution. The exception is caught so the most recent output will be dumped to dashboard",
        )

        image_x = st.sidebar.number_input(
            "Xdim", value=defaults["Xdim"], help="Width of output image, in pixels"
        )
        image_y = st.sidebar.number_input(
            "ydim", value=defaults["ydim"], help="Height of output image, in pixels"
        )
        set_seed = st.sidebar.checkbox(
            "Set seed",
            value=defaults["set_seed"],
            help="Check to set random seed for reproducibility. Will add option to specify seed",
        )

        seed_widget = st.sidebar.empty()
        if set_seed is True:
            # Use text_input as number_input relies on JS
            # which can't natively handle large numbers
            # torch.seed() generates int w/ 19 or 20 chars!
            seed_str = seed_widget.text_input(
                "Seed", value=str(defaults["seed"]), help="Random seed to use"
            )
            try:
                seed = int(seed_str)
            except ValueError as e:
                st.error("seed input needs to be int")
        else:
            seed = None

        use_custom_starting_image = st.sidebar.checkbox(
            "Use starting image",
            value=defaults["use_starting_image"],
            help="Check to add a starting image to the network",
        )

        starting_image_widget = st.sidebar.empty()
        if use_custom_starting_image is True:
            init_image = starting_image_widget.file_uploader(
                "Upload starting image",
                type=["png", "jpeg", "jpg"],
                accept_multiple_files=False,
                help="Starting image for the network, will be resized to fit specified dimensions",
            )
            # Convert from UploadedFile object to PIL Image
            if init_image is not None:
                init_image: Image.Image = Image.open(init_image).convert(
                    "RGB"
                )  # just to be sure
        else:
            init_image = None

        use_image_prompts = st.sidebar.checkbox(
            "Add image prompt(s)",
            value=defaults["use_image_prompts"],
            help="Check to add image prompt(s), conditions the network similar to the text prompt",
        )

        image_prompts_widget = st.sidebar.empty()
        if use_image_prompts is True:
            image_prompts = image_prompts_widget.file_uploader(
                "Upload image prompts(s)",
                type=["png", "jpeg", "jpg"],
                accept_multiple_files=True,
                help="Image prompt(s) for the network, will be resized to fit specified dimensions",
            )
            # Convert from UploadedFile object to PIL Image
            if len(image_prompts) != 0:
                image_prompts = [Image.open(i).convert("RGB") for i in image_prompts]
        else:
            image_prompts = []

        continue_prev_run = st.sidebar.checkbox(
            "Continue previous run",
            value=defaults["continue_prev_run"],
            help="Use existing image and existing weights for the next run. If yes, ignores 'Use starting image'",
        )
        submitted = st.form_submit_button("Run!")
        # End of form

    status_text = st.empty()
    status_text.text("Pending input prompt")
    step_progress_bar = st.progress(0)

    im_display_slot = st.empty()
    vid_display_slot = st.empty()
    debug_slot = st.empty()

    if "prev_im" in st.session_state:
        im_display_slot.image(
            st.session_state["prev_im"], caption="Output image", output_format="PNG"
        )

    with st.beta_expander("Expand for README"):
        with open("README.md", "r") as f:
            # description = f.read()
            # Preprocess links to redirect to github
            # Thank you https://discuss.streamlit.io/u/asehmi, works like a charm!
            # ref: https://discuss.streamlit.io/t/image-in-markdown/13274/8
            readme_lines = f.readlines()
            readme_buffer = []
            images = ["docs/ui.jpeg", "docs/four-seasons-20210808.png"]
            for line in readme_lines:
                readme_buffer.append(line)
                for image in images:
                    if image in line:
                        st.markdown(" ".join(readme_buffer[:-1]))
                        st.image(
                            f"https://raw.githubusercontent.com/tnwei/vqgan-clip-app/main/{image}"
                        )
                        readme_buffer.clear()
            st.markdown(" ".join(readme_buffer))

        # st.write(description)

    if submitted:
        # debug_slot.write(st.session_state) # DEBUG
        status_text.text("Loading weights ...")
        generate_image(
            # Inputs
            text_input=text_input,
            vqgan_ckpt=radio,
            num_steps=num_steps,
            image_x=int(image_x),
            image_y=int(image_y),
            seed=int(seed) if set_seed is True else None,
            init_image=init_image,
            image_prompts=image_prompts,
            continue_prev_run=continue_prev_run,
        )
        vid_display_slot.video("temp.mp4")
        # debug_slot.write(st.session_state) # DEBUG
