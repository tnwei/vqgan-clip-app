from flask import Flask, render_template, send_from_directory
from gallery_utils import RunResults
from pathlib import Path
import argparse


def update_runs(fdir):
    runs = []
    # Load each run as ModelRun objects
    # Loading the latest ones first
    for i in sorted(fdir.iterdir(), reverse=True):
        # If is a folder and contains images and metadata
        if i.is_dir() and (i / "details.json").exists() and (i / "output.PNG").exists():
            try:
                runs.append(RunResults(i))
            except Exception as e:
                print(f"Skipped {i} due to raised exception {e}")
                pass
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
    args = parser.parse_args()

    fdir = Path(args.path)
    runs = update_runs(fdir)

    app = Flask(
        __name__,
        # Hack to allow easy access to images
        # Else typically this stuff needs to be put in a static/ folder!
        static_url_path="",
        static_folder="",
    )

    @app.route("/")
    def home():
        runs = update_runs(fdir)  # Updates new results when refreshed
        return render_template("index.html", runs=runs, fdir=fdir)

    @app.route("/findurl")
    def findurl(path, filename):
        return send_from_directory(path, filename)

    app.run(debug=True)
