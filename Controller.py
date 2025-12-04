import zlib
import cv2

from functools import partial
from typing import Callable

from file_analysis import StegoFileInspector
from rs_analysis import RSAnalyzer

class Controller:

    def handle_detect(
        self,
        filepath: str
        ):
        print(f"filepath={filepath}")
        try:
            extension = (filepath.split('.'))[1]
            if extension not in ["png", "jpg", "jpeg"]:
                raise ValueError(f"Unable to support decoding for {extension} files!")
            
            # Perform file analysis
            file_analyzer = StegoFileInspector(filepath)
            anomaly_report = file_analyzer.get_anomaly_report()

            # Run RS Analysis to obtain confidence score
            rs_analyzer = RSAnalyzer(block_size=16, backend="python")
            rs_map = rs_analyzer.analyze(filepath)
            rs_confidence = rs_analyzer.stego_confidence(rs_map)
            print("RS confidence: " + str(rs_confidence))

            # Obtain RS Analysis heatmap(highlights suspicious areas)
            rs_heatmap_fig = rs_analyzer.make_heatmap_figure(rs_map)

            img_grayscale = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            rs_overlay_fig = rs_analyzer.make_old_style_overlay_figure(img_grayscale, rs_map)

            # Save RS Analysis heatmaps for use in Latex
            rs_heatmap_fig.savefig("rs_heatmap.pdf", dpi=300, bbox_inches="tight")
            rs_overlay_fig.savefig("rs_overlay.pdf", dpi=300, bbox_inches="tight")
            # Then in Latex: \includegraphics[width=0.9\linewidth]{rs_overlay.pdf}

        except Exception as e:
            exception = e