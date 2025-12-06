import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from typing import Callable
from cnn.steganography_detector import SteganographyDetector

from file_analysis import StegoFileInspector
from statistical_analysis.rs_analysis import RSAnalyzer
from statistical_analysis.high_pass_residual import HighPassResidualSteganalysis

from latex_report.latex_report_generator import LatexReportGenerator

class Controller:
    def __init__(self):
        self.cnn_detector = SteganographyDetector(image_size=(128, 128))
        self.cnn_detector.model = tf.keras.models.load_model("cnn/model/steg_detector.h5")
        pass

    def get_file_anomaly_warnings(self, filepath: str):
        file_analyzer = StegoFileInspector(filepath)
        return file_analyzer.get_anomaly_report()
    
    def get_rs_analysis_artifacts(self, filepath: str):
        rs_analyzer = RSAnalyzer(block_size=16, backend="python")
        rs_map = rs_analyzer.analyze(filepath)
        rs_confidence = rs_analyzer.stego_confidence(rs_map)

        # Obtain RS Analysis heatmap(highlights suspicious areas)
        rs_heatmap_fig = rs_analyzer.make_heatmap_figure(rs_map)

        img_grayscale = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        rs_overlay_fig = rs_analyzer.make_old_style_overlay_figure(img_grayscale, rs_map)

        rs_heatmap_fig.savefig("latex_report/rs_heatmap.pdf", dpi=300, bbox_inches="tight")
        rs_overlay_fig.savefig("latex_report/rs_overlay.pdf", dpi=300, bbox_inches="tight")

        # Need to close the figs to avoid error
        plt.close(rs_heatmap_fig)
        plt.close(rs_overlay_fig)

        return rs_confidence
    
    def get_high_pass_residual_artifacts(self, filepath: str):
        hpr_analyzer = HighPassResidualSteganalysis(window_size=8)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        hpr_confidence, residual_fig, var_map, heatmap = hpr_analyzer.analyze(filepath)

        fig = hpr_analyzer.make_figure(img, residual_fig, var_map, heatmap)
        fig.savefig("latex_report/high_pass_analysis.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

        return hpr_confidence
    
    def get_cnn_confidence_score(self, filepath: str):
        self.cnn_detector.model = tf.keras.models.load_model("cnn/model/steg_detector.h5")

        return self.cnn_detector.predict_image(filepath)

    def handle_detect(
        self,
        filepath: str,
        report_output_path: str
        ):
        print(f"filepath={filepath}")
        try:
            extension = (filepath.split('.'))[1]

            if extension not in ["png", "jpg", "jpeg"]:
                raise ValueError(f"Unable to support decoding for {extension} files!")
            
            # Perform file analysis
            file_anomaly_warnings = self.get_file_anomaly_warnings(filepath)
            print(f"We have {len(file_anomaly_warnings)} anomalies!")

            # Run RS Analysis to obtain confidence score and heatmaps
            rs_confidence = self.get_rs_analysis_artifacts(filepath)
            print(f"rs_confidence = {rs_confidence}")

            # Run High Pass Residual Analysis to obtain confidence score and heatmaps
            hpr_confidence = self.get_high_pass_residual_artifacts(filepath)
            print(f"hpr_confidence = {hpr_confidence}")

            # Get the confidence score from the neural network
            cnn_confidence = self.get_cnn_confidence_score(filepath)
            print(f"Steganography detected: (confidence: {cnn_confidence:.2f})")
            
            confidence_average = rs_confidence * 0.2 + hpr_confidence * 0.4 + cnn_confidence * 0.4

            report_generator = LatexReportGenerator("latex_report/report_template.tex")
            report_generator.generate_report(filepath, confidence_average, file_anomaly_warnings)
            report_generator.compile_pdf("latex_report/report.tex", report_output_path)
            

        except Exception as e:
            exception = e
            print(exception)