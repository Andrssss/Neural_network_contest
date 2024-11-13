# logger.py
import logging
import sys
import codecs
from datetime import datetime
import os


def setup_logger(log_dir="log"):
    """Beállítja a logger-t fájlba és konzolra történő íráshoz."""

    # Létrehozza a log mappát, ha nem létezik
    os.makedirs(log_dir, exist_ok=True)

    # Dátum/idő alapú log fájl
    date_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    log_file_path = f"{log_dir}/log_{date_str}.txt"

    # UTF-8 kimenet biztosítása konzolhoz
    class Utf8StreamHandler(logging.StreamHandler):
        def __init__(self, stream=None):
            if stream is None:
                stream = sys.stdout
            super().__init__(stream=codecs.getwriter("utf-8")(stream.buffer))

    # Log konfiguráció fájlba és konzolra
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file_path, encoding="utf-8"),  # Fájl log
            Utf8StreamHandler()  # Konzol log
        ]
    )
