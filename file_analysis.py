import os
import struct
from PIL import Image
import exifread

class StegoFileInspector:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.filetype = os.path.splitext(filepath)[1].lower()
        self.warnings = []

    def _check_file_size(self):
        size = os.path.getsize(self.filepath)
        if size < 1024:
            self.warnings.append(f"File size unusually small ({size} bytes).")
        elif size > 10 * 1024 * 1024:
            self.warnings.append(f"File size unusually large ({size/1024/1024:.2f} MB).")

    def _check_headers(self):
        try:
            with open(self.filepath, "rb") as f:
                header = f.read(8)
                if self.filetype == ".png" and header != b'\x89PNG\r\n\x1a\n':
                    self.warnings.append("Invalid PNG header.")
                elif self.filetype in [".jpg", ".jpeg"]:
                    f.seek(0)
                    soi = f.read(2)
                    if soi != b'\xFF\xD8':
                        self.warnings.append("Invalid JPEG header (missing SOI marker).")
        except Exception as e:
            self.warnings.append(f"Header check failed: {e}")

    def _check_metadata(self):
        if self.filetype not in [".jpg", ".jpeg", ".tiff"]:
            return
        try:
            with open(self.filepath, "rb") as f:
                tags = exifread.process_file(f, details=False)
            if not tags:
                self.warnings.append("No EXIF metadata found (possible stripping or modification), but low priority.")
            else:
                # Check for anomalies
                if "Image Make" not in tags or "Image Model" not in tags:
                    self.warnings.append("Incomplete EXIF metadata (missing camera info).")
                if "EXIF DateTimeOriginal" not in tags:
                    self.warnings.append("Missing original timestamp in EXIF data.")
        except Exception as e:
            self.warnings.append(f"EXIF check failed: {e}")

    def _check_png_chunks(self):
        try:
            with open(self.filepath, "rb") as f:
                f.seek(8)  # skip PNG signature
                while True:
                    data = f.read(8)
                    if len(data) < 8:
                        break
                    length, chunk_type = struct.unpack(">I4s", data)
                    chunk_data = f.read(length)
                    crc_read = f.read(4)
                    if len(crc_read) < 4:
                        break
                    crc_calc = struct.pack(">I", (zlib.crc32(chunk_type + chunk_data) & 0xffffffff))
                    if crc_calc != crc_read:
                        self.warnings.append(f"CRC mismatch in chunk {chunk_type.decode('ascii')}.")
                    # Check for nonstandard chunks
                    if chunk_type not in [b'IHDR', b'PLTE', b'IDAT', b'IEND', b'tEXt', b'zTXt', b'iTXt']:
                        self.warnings.append(f"Nonstandard PNG chunk found: {chunk_type.decode('ascii')}")
        except Exception as e:
            self.warnings.append(f"PNG chunk analysis failed: {e}")
    
    def analyze(self):
        self._check_file_size()
        self._check_headers()
        self._check_metadata()
        if self.filetype == ".png":
            self._check_png_chunks()
        return self.report()

    def report(self):
        if not self.warnings:
            return f"[{self.filename}] No anomalies detected."
        else:
            report = f"[{self.filename}] Potential issues found:\n"
            for w in self.warnings:
                report += f"  - {w}\n"
            return report
        

file_analyzer = StegoFileInspector("foid.jpg")
file_analyzer.analyze()
print(file_analyzer.report())