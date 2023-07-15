import sys 
from src.logger import logging

def error_message_detail(error, error_detail:sys):

    _,_, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error_message = f'Error occured in the python script named : {filename} , at line no. : {line_no} , error details : {error} '

    return error_message

class Custom_Exception(Exception):

    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
          error= error_message,
          error_detail=error_detail  
        )

    def __str__(self) -> str:
        return self.error_message