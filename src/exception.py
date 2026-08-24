from src.logger import logging
import sys


def error_message_details(error,error_details:sys):
    _,_,exc_tb = error_details.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message =( f"error message occure python script name: {file_name}"
                    f"at line number: {line_number}"
                    f"with message: {str(error)}")

    return error_message

class CustomException:
    def __init__(self,error_message,erorr_details:sys):
        super().__init__(error_message)
        self.erorr_message_details(error_message,erorr_details=erorr_details)


    def __str__(self):
        return self.erorr_message