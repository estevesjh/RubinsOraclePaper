"""Compile the twilight forecast report LaTeX paper to PDF.

Usage:
    python compile_paper.py          # Compile once
    python compile_paper.py --clean  # Clean auxiliary files first
    python compile_paper.py --open   # Compile and open PDF
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Paths
DOCS_PATH = Path(__file__).parent.parent / "docs"
TEX_FILE = DOCS_PATH / "twilight_forecast_report.tex"
PDF_FILE = DOCS_PATH / "twilight_forecast_report.pdf"

# Auxiliary file extensions to clean
AUX_EXTENSIONS = [
    ".aux", ".log", ".out", ".toc", ".lof", ".lot",
    ".fls", ".fdb_latexmk", ".synctex.gz", ".bbl", ".blg"
]


def clean_aux_files():
    """Remove auxiliary LaTeX files."""
    print("Cleaning auxiliary files...")
    count = 0
    for ext in AUX_EXTENSIONS:
        for f in DOCS_PATH.glob(f"*{ext}"):
            f.unlink()
            count += 1
    print(f"  Removed {count} files")


def compile_latex(tex_file: Path, runs: int = 2) -> bool:
    """Compile LaTeX file to PDF.

    Args:
        tex_file: Path to .tex file
        runs: Number of pdflatex runs (2 for references)

    Returns:
        True if successful
    """
    print(f"Compiling {tex_file.name}...")

    for i in range(runs):
        print(f"  Pass {i+1}/{runs}...")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_file.name],
            cwd=DOCS_PATH,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"  ERROR: pdflatex failed!")
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            return False

    return True


def open_pdf(pdf_file: Path):
    """Open PDF in default viewer."""
    import platform

    if platform.system() == "Darwin":  # macOS
        subprocess.run(["open", pdf_file])
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", pdf_file])
    elif platform.system() == "Windows":
        subprocess.run(["start", pdf_file], shell=True)


def main():
    parser = argparse.ArgumentParser(description="Compile LaTeX paper to PDF")
    parser.add_argument("--clean", action="store_true", help="Clean auxiliary files first")
    parser.add_argument("--open", action="store_true", help="Open PDF after compilation")
    args = parser.parse_args()

    print("=" * 60)
    print("COMPILE TWILIGHT FORECAST REPORT")
    print("=" * 60)

    # Check tex file exists
    if not TEX_FILE.exists():
        print(f"ERROR: {TEX_FILE} not found!")
        sys.exit(1)

    # Clean if requested
    if args.clean:
        clean_aux_files()

    # Compile
    success = compile_latex(TEX_FILE)

    if success:
        print(f"\nSUCCESS: {PDF_FILE}")

        if args.open:
            print("Opening PDF...")
            open_pdf(PDF_FILE)
    else:
        print("\nFAILED: Check LaTeX errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
