import argparse
import json
from pathlib import Path
from typing import Union

import yaml
from flask import Flask, render_template, request, send_from_directory
from PIL import Image


class RunResults:
    """
    Store run output and metadata as a class for ease of use.
    """

    def __init__(self, fdir: Union[str, Path]):
        self.fdir: str = Path(fdir)  # exactly as given
        self.absfdir: Path = Path(fdir).resolve()  # abs
        files_available = [i.name for i in self.fdir.glob("*")]

        # The most important info is the final image and the metadata
        # Leaving an option to delete videos to save space
        if (
            "details.txt" not in files_available
            and "details.json" not in files_available
        ):
            raise ValueError(
                f"fdir passed has neither details.txt or details.json: {fdir}"
            )

        if "output.PNG" not in files_available:
            raise ValueError(f"fdir passed contains no output.PNG: {fdir}")

        self.impath = (self.fdir / "output.PNG").as_posix()

        if "anim.mp4" in files_available:
            self.animpath = (self.fdir / "anim.mp4").as_posix()
        else:
            self.animpath = None
            print(f"fdir passed contains no anim.mp4: {fdir}")

        if "init-image.JPEG" in files_available:
            self.initimpath = (self.fdir / "init-image.JPEG").as_posix()
        else:
            self.initimpath = None

        self.impromptspath = [i.as_posix() for i in self.fdir.glob("image-prompt*")]
        if len(self.impromptspath) == 0:
            self.impromptspath = None

        if "details.txt" in files_available:
            self.detailspath = (self.fdir / "details.txt").as_posix()
        elif "details.json" in files_available:
            self.detailspath = (self.fdir / "details.json").as_posix()

        with open(self.detailspath, "r") as f:
            self.details = json.load(f)

        # Preserving the filepaths as I realize might be calling
        # them through Jinja in HTML instead of within the app
        # self.detailshtmlstr = markdown2.markdown(
        #     "```" + json.dumps(self.details, indent=4) + "```",
        #     extras=["fenced-code-blocks"]
        # )
        self.detailshtmlstr = yaml.dump(self.details)

        # Replace with line feed and carriage return
        # ref: https://stackoverflow.com/a/39325879/13095028
        self.detailshtmlstr = self.detailshtmlstr.replace("\n", "<br>")
        # Edit: Using preformatted tag <pre>, solved!

        self.im = Image.open(self.fdir / "output.PNG").convert(
            "RGB"
        )  # just to be sure the format is right


def update_runs(fdir, runs):
    existing_run_folders = [i.absfdir.name for i in runs]
    # Load each run as ModelRun objects
    # Loading the latest ones first
    for i in sorted(fdir.iterdir()):
        # If is a folder and contains images and metadata
        if (
            i.is_dir()
            and (i / "details.json").exists()
            and (i / "output.PNG").exists()
            and i.name not in existing_run_folders
        ):
            try:
                runs.insert(0, RunResults(i))
            except Exception as e:
                print(f"Skipped {i} due to raised exception {e}")
    return runs


if __name__ == "__main__":
    # Select output dir
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        help="Path to output folder, should contain subdirs of individual runs",
        nargs="?",
        default="./output/",
    )
    parser.add_argument(
        "-n",
        "--numitems",
        help="Number of items per page",
        default=24,  # multiple of three since the dashboard has three panels
    )
    parser.add_argument(
        "--kiosk",
        help="Omit showing run details on dashboard",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    fdir = Path(args.path)
    runs = update_runs(fdir, [])

    app = Flask(
        __name__,
        # Hack to allow easy access to images
        # Else typically this stuff needs to be put in a static/ folder!
        static_url_path="",
        static_folder="",
    )

    @app.route("/")
    def home():
        # startidx = request.args.get('startidx')
        # endidx = request.args.get('endidx')

        # Pagenum starts at 1
        page = request.args.get("page")
        page = 1 if page is None else int(page)

        # startidx = 1 if startidx is None else int(startidx)
        # endidx = args.numitems if endidx is None else int(endidx)
        # print("startidx, endidx: ", startidx, endidx)
        global runs
        runs = update_runs(fdir, runs)  # Updates new results when refreshed
        num_pages = (len(runs) // args.numitems) + 1

        page_labels = {}
        for i in range(0, num_pages):
            page_labels[i + 1] = dict(
                start=i * args.numitems + 1, end=(i + 1) * args.numitems
            )

        return render_template(
            "index.html",
            runs=runs,
            startidx=page_labels[page]["start"],
            endidx=page_labels[page]["end"],
            page=page,
            fdir=fdir,
            page_labels=page_labels,
            kiosk=args.kiosk,
        )

    @app.route("/findurl")
    def findurl(path, filename):
        return send_from_directory(path, filename)

    app.run(debug=False)
