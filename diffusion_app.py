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

import imageio
import numpy as np
from diffusion_logic import (
    # CLIPGuidedDiffusion256HQ,
    # CLIPGuidedDiffusion512HQ,
    # CLIPGuidedDiffusion512HQUncond,
    CLIPGuidedDiffusion,
    DIFFUSION_METHODS_AND_WEIGHTS,
)


def generate_image(
    diffusion_weights: str, prompt: str, seed=0, num_steps=500, continue_prev_run=True
) -> None:

    ### Init -------------------------------------------------------------------
    run = CLIPGuidedDiffusion(
        prompt=prompt,
        weights=diffusion_weights,
        seed=seed,
        num_steps=num_steps,
        continue_prev_run=continue_prev_run,
    )

    # Generate random run ID
    # Used to link runs linked w/ continue_prev_run
    # ref: https://stackoverflow.com/a/42703382/13095028
    # Use URL and filesystem safe version since we're using this as a folder name
    run_id = st.session_state["run_id"] = base64.urlsafe_b64encode(
        os.urandom(6)
    ).decode("ascii")

    run_start_dt = datetime.datetime.now()

    ### Load model -------------------------------------------------------------
    if (
        continue_prev_run
        and ("model" in st.session_state)
        and ("clip_model" in st.session_state)
        and ("diffusion" in st.session_state)
    ):
        run.load_model(
            prev_model=st.session_state["model"],
            prev_diffusion=st.session_state["diffusion"],
            prev_clip_model=st.session_state["clip_model"],
        )
    else:
        (
            st.session_state["model"],
            st.session_state["diffusion"],
            st.session_state["clip_model"],
        ) = run.load_model(model_file_loc="assets/" + diffusion_method)

    ### Model init -------------------------------------------------------------
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

            ims = run.iterate()
            im = ims[0]

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

        im.save(runoutputdir / "output.PNG", format="PNG")
        shutil.copy("temp.mp4", runoutputdir / "anim.mp4")

        with open(runoutputdir / "details.json", "w") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "ckpt": diffusion_method,
                    "num_steps": step_counter,
                    "planned_num_steps": num_steps,
                    "text_input": prompt,
                    "continue_prev_run": continue_prev_run,
                    "seed": seed,
                    "Xdim": imsize,
                    "ydim": imsize,
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

        # Save to output folder if run completed
        runoutputdir = outputdir / (
            run_start_dt.strftime("%Y%m%dT%H%M%S") + "-" + run_id
        )
        runoutputdir.mkdir()

        im.save(runoutputdir / "output.PNG", format="PNG")
        shutil.copy("temp.mp4", runoutputdir / "anim.mp4")

        with open(runoutputdir / "details.json", "w") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "ckpt": diffusion_method,
                    "num_steps": step_counter,
                    "planned_num_steps": num_steps,
                    "text_input": prompt,
                    "continue_prev_run": continue_prev_run,
                    "seed": seed,
                    "Xdim": imsize,
                    "ydim": imsize,
                    "start_time": run_start_dt.strftime("%Y%m%dT%H%M%S"),
                    "end_time": datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
                },
                f,
                indent=4,
            )

        status_text.text("Done!")  # End of run


if __name__ == "__main__":
    outputdir = Path("output")
    if not outputdir.exists():
        outputdir.mkdir()

    st.set_page_config(page_title="CLIP guided diffusion playground")
    st.title("CLIP guided diffusion playground")

    with open("diffusion_README.md", "r") as f:
        description = f.read()
    st.write(description)

    # Start of input form
    with st.form("form-inputs"):
        # Only element not in the sidebar, but in the form

        text_input = st.text_input(
            "Text prompt",
            help="CLIP-guided diffusion will generate an image that best fits the prompt",
        )

        diffusion_method = st.sidebar.radio(
            "Method",
            list(DIFFUSION_METHODS_AND_WEIGHTS.keys()),
            index=0,
            help="Choose diffusion image generation method, corresponding to the notebooks in Eleuther's repo",
        )

        if diffusion_method.startswith("256"):
            image_size_notice = st.sidebar.text("Image size: fixed to 256x256")
            imsize = 256
        elif diffusion_method.startswith("512"):
            image_size_notice = st.sidebar.text("Image size: fixed to 512x512")
            imsize = 512

        set_seed = st.sidebar.checkbox(
            "Set seed",
            value=0,
            help="Check to set random seed for reproducibility. Will add option to specify seed",
        )
        num_steps = st.sidebar.number_input(
            "Num steps",
            value=1000,
            min_value=0,
            max_value=None,
            step=1,
            # help="Specify -1 to run indefinitely. Use Streamlit's stop button in the top right corner to terminate execution. The exception is caught so the most recent output will be dumped to dashboard",
        )

        seed_widget = st.sidebar.empty()
        if set_seed is True:
            seed = seed_widget.number_input("Seed", value=0, help="Random seed to use")
        else:
            seed = None

        continue_prev_run = st.sidebar.checkbox(
            "Skip init if models are loaded",
            value=True,
            help="Skips lengthy model init",
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

    # with st.beta_expander("Expand for README"):
    #     with open("README.md", "r") as f:
    #         # description = f.read()
    #         # Preprocess links to redirect to github
    #         # Thank you https://discuss.streamlit.io/u/asehmi, works like a charm!
    #         # ref: https://discuss.streamlit.io/t/image-in-markdown/13274/8
    #         readme_lines = f.readlines()
    #         readme_buffer = []
    #         images = ["docs/ui.jpeg", "docs/four-seasons-20210808.png"]
    #         for line in readme_lines:
    #             readme_buffer.append(line)
    #             for image in images:
    #                 if image in line:
    #                     st.markdown(" ".join(readme_buffer[:-1]))
    #                     st.image(
    #                         f"https://raw.githubusercontent.com/tnwei/vqgan-clip-app/main/{image}"
    #                     )
    #                     readme_buffer.clear()
    #         st.markdown(" ".join(readme_buffer))

    # st.write(description)

    if submitted:
        # debug_slot.write(st.session_state) # DEBUG
        status_text.text("Loading weights ...")
        generate_image(
            diffusion_weights=diffusion_method,
            prompt=text_input,
            seed=seed,
            num_steps=num_steps,
            continue_prev_run=continue_prev_run,
        )
        vid_display_slot.video("temp.mp4")
        # debug_slot.write(st.session_state) # DEBUG
