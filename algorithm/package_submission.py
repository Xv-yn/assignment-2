"""Package reproducible source and an empty data directory, excluding datasets."""

import argparse
from pathlib import Path
import zipfile

from .data import ROOT


def package(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in (ROOT / "algorithm").rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".md"))
    files += [ROOT / "README.md", ROOT / "requirements.txt"]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data/", "")
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as archive:
        if any(p.startswith("data/") and p != "data/" for p in archive.namelist()):
            raise AssertionError("Submission data directory is not empty")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Use the assignment's group/student-ID naming convention")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".zip":
        parser.error("Output must end in .zip")
    print(package(args.output))


if __name__ == "__main__":
    main()
