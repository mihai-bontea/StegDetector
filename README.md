# StegDetector

Steganalysis is the science of detecting the presence of hidden messages in media. There are multiple steganalysis techniques, which usually look for the
artifacts created by the message hiding process. This tool combines metadata analysis, statistical analysis and neural networks, for the purpose of identifying
whether or not an image contains embedded messages.

## File Metadata Anomalies

Sometimes the process of steganography leaves other, more noticeable traces besides the noise in the image itself. The image has been scanned for such traces:
structural, metadata, and container anomalies that might indicate tampering or hidden data.

Examples include unusual file size, invalid file headers, missing or strange EXIF metadata, and corrupted or suspicious PNG chunks. While these are not a direct proof of
steganography, they are heuristic red flags.

## RS Analysis

RS Analysis is a statistical steganalysis technique designed to detect data hidden by LSB (least-significant bit) embedding in images. It is based on the fact that in a
non-stego image, there are certain regular statistical relationships between the different bit-planes and among neighboring pixels. By embedding a message (modifying
LSBs), these correlations are disturbed, and RS Analysis can detect these disturbances. It is effective when the cover image is a "natural image" (e.g. photos with 
natural textures and smooth areas) because those have the spatial correlations that RS exploits. In addition, the amount of data hidden needs to be sufficiently high,
since small payloads may produce a disturbance that is too subtle to reliably detect.

## High-Pass Residual Analysis

High-pass residual analysis refers to a class of statistical steganalysis techniques that apply high-pass filters(or residual filters) to an image, in order to suppress
the "normal content" (smooth, large-scale textures) and amplify the small perturbations or "noise" caused by steganography use. This results in a residual image:
a two-dimensional array showingdifferences, edges, and high-frequency detail. Hidden data often shows up in these residuals more clearly than in the original image. For this 
reason, these residuals are used by many modern steganalysis methods.

## CNN-Based Steganography Detection

A convolutional neural network (CNN) was used to provide an additional learned-based estimate of steganographic embedding probability.
The model has been trained on the BOSSBase(Break Our Steganographic System Base), which is one of the most important and widely used benchmark datasets
in steganography and steganalysis research. It consists of 18,000 grayscale images.

A second model has also been trained on the high-pass residual images obtained based on the regular dataset.

## LaTeX Report

With information taken from the methods above, including confidence scores and heatmaps with suspicious areas highlighted, a LaTeX report is generated, and converted to pdf. This contains the final verdict, which is obtained through a weighted mean of the confidence scores of all methods used.
