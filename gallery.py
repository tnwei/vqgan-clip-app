from flask import Flask, render_template, send_from_directory, request
from gallery_utils import RunResults
from pathlib import Path
import argparse


def update_runs(fdir, runs):
    existing_run_folders = [i.absfdir.name for i in runs]
    # Load each run as ModelRun objects
    # Loading the latest ones first
    for i in sorted(fdir.iterdir(), reverse=True):
        # If is a folder and contains images and metadata
        if (
            i.is_dir()
            and (i / "details.json").exists()
            and (i / "output.PNG").exists()
            and i.name not in existing_run_folders
        ):
            try:
                runs.append(RunResults(i))
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
