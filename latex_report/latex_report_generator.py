import subprocess
import shutil
from pathlib import Path
from typing import Callable

class LatexReportGenerator:
    """
    Handles:
    - Filling a LaTeX template with values
    - Formatting metadata warnings
    - Compiling the final PDF using pdflatex
    """

    def __init__(self, template_path="report_template.tex"):
        self.template_path = template_path

        if not shutil.which("pdflatex"):
            raise RuntimeError(
                "pdflatex not found. Please install MiKTeX / TeX Live / MacTeX."
            )

    @staticmethod
    def _escape_latex(text: str) -> str:
        """
        Escapes characters that break LaTeX.
        """
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
            "\\": r"\textbackslash{}",
        }

        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    def _format_warnings(self, warnings: list[str]) -> str:
        """
        Converts a list of anomaly strings into a LaTeX bullet list.
        """
        if not warnings:
            return r"\textbf{No file metadata anomalies were detected.}"

        lines = ["\\begin{itemize}"]
        for w in warnings:
            safe = self._escape_latex(w)
            lines.append(f"  \\item {safe}")
        lines.append("\\end{itemize}")

        return "\n".join(lines)

    def generate_report(
        self,
        image_name: str,
        confidence: float,
        file_warnings: list[str],
        output_tex="latex_report/report.tex",
    ):
        """
        Fills the LaTeX template with values and writes report.tex
        """

        with open(self.template_path, "r") as f:
            template = f.read()

        warnings_block = self._format_warnings(file_warnings)

        filled = template.replace("{{IMAGE_NAME}}", self._escape_latex(image_name))
        filled = filled.replace("{{CONFIDENCE}}", f"{confidence:.4f}")
        filled = filled.replace("{{FILE_WARNINGS}}", warnings_block)

        with open(output_tex, "w") as f:
            f.write(filled)

        return output_tex

    def compile_pdf(self, tex_file, output_path):
        tex_path = Path(tex_file).resolve()

        # Compile LaTeX to PDF
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=str(tex_path.parent),
            check=True
        )

        generated_pdf = tex_path.with_suffix(".pdf")

        save_path = Path(output_path)
        if save_path.is_dir():
            save_path = save_path / generated_pdf.name

        shutil.move(str(generated_pdf), str(save_path))

        return save_path
