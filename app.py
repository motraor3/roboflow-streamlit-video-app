import streamlit as st
import requests
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

import asyncio
import logging
import logging.handlers
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple
import time
import video_stream

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

### setting up the logger to log messages created by the script to help debug in main()
logger = logging.getLogger(__name__)


def main():
    """
    Main part of the script that defines the main method and calls the logo detection.
    Also sets up a logger for debugging purposes.
    """

    process_video = st.button('Process Video')

    while process_video:
        st.header("Inference on a Video Feed with the Roboflow API")

        video_stream.video_dashboard()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f" {thread.name} ({thread.ident})")


### the main program where we call the main method
if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
