import os
import sys


class AppException(Exception):
    def _init_(self, error_message: Exception, error_detail: sys):
        """
        :param error_message: error message in string format
        """
        super()._init_(error_message)
        self.error_message = AppException.error_message_detail(error_message, error_detail=error_detail)

    @staticmethod
    def error_message_detail(error:Exception, error_detail:sys):
        """
        error: Exception object raise from module
        error_detail: is sys module contains detail information about system execution information.
        """
        _, _, exc_tb = error_detail.exc_info()
        #extracting file name from exception traceback
        file_name = exc_tb.tb_frame.f_code.co_filename 

        #preparing error message
        error_message = f"Error occurred python script name [{file_name}]" \
                        f" line number [{exc_tb.tb_lineno}] error message [{error}]."

        return error_message

    def _repr_(self):
        """
        Formating object of AppException
        """
        return AppException._name.str_()

    def _str_(self):
        """
        Formating how a object should be visible if used in print statement.
        """
        return self.error_message