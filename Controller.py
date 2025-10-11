import zlib

from functools import partial
from typing import Callable

class Controller:

    def handle_detect(
        self,
        filepath: str,
        ):
        exception = None
        decompressed_message = None
        print(f"filepath={filepath}")
        try:
            # Determine the carrier's extension and call the appropriate decoding function
            extension = (filepath.split('.'))[1]
            match extension:
                case "png":
                    print("Png file")
                case 'jpg':
                    print("Jng file")
                case default:
                    raise ValueError(f"Unable to support decoding for {extension} files!")
            
            # Run the detection actions

        except Exception as e:
            exception = e