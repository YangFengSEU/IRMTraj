import logging
import time
import os
import sys
def logger_init(file_path, file_name,log_level = logging.INFO, only_file = False):
    timenow = time.strftime('%Y-%m-%d-%H:%M',time.localtime())
    file_path = file_path+'/'+timenow
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    log_path = os.path.join(file_path,file_name)
    format_log = "%(asctime)s-%(levelname)s-%(message)s"

    if only_file:
        logging.basicConfig(filename = log_path, level=log_level, format = format_log, datefmt='%Y-%m-%d-%H:%M:%S')
    else:
        logging.basicConfig(level=log_level, 
                            format = format_log,
                             datefmt='%Y-%m-%d-%H:%M:%S',
                             handlers=[logging.FileHandler(log_path),
                                       logging.StreamHandler(sys.stdout)])        

