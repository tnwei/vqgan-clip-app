from typing import Union
from pathlib import Path
import json
from PIL import Image
import yaml


class RunResults:
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
