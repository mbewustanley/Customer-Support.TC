import sys
from src.utils.logger import get_logger


def error_message_detail(error: Exception, error_detail: sys):
    # Extract the traceback information from the error_detail object
    _, _, exc_tb = error_detail.exc_info()

    # Get the filename and line number where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{str(error)}]"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)


    def __str__(self):
        return self.error_message
    