import os
import logging
from datetime import datetime

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR, 
    f"pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
logging.basicConfig(filename=LOG_FILE, level=logging.INFO)



def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    
    logger = logging.getLogger(name)

    #prevent adding multiple handlers to the logger if it already has handlers
    if not logger.handlers:

        # Create a formatter and set it for the console handler and file handler
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Create a file handler
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Add the console and file handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger