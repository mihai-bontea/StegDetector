import subprocess
import shutil


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

    # -------------------------------
    # Internal helpers
    # -------------------------------

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

    # -------------------------------
    # Public API
    # -------------------------------

    def generate_report(
        self,
        image_name: str,
        confidence: float,
        file_warnings: list[str],
        output_tex="report.tex",
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

    def compile_pdf(self, tex_file="report.tex"):
        """
        Compiles the LaTeX file into a PDF.
        """
        subprocess.run(
            ["pdflatex", tex_file],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        pdf_file = tex_file.replace(".tex", ".pdf")
        return pdf_file
