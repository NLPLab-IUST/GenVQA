import datetime
from src.constants import LOGS_DIR
import os
class Logger:
    
    def __init__(self, logs_dir):
        self.logs_dir = logs_dir

    def log(self, module, message):
        os.makedirs(self.logs_dir, exist_ok=True)
        logs_path = self.logs_dir + f"/{module}.logs"
        message = f"[{datetime.datetime.now()}] " + message + "\n"
        if os.path.exists(logs_path):
            method = 'a'
        else :
            method = 'w'
        with open(logs_path, method) as f:
                f.write(message)

Instance  = Logger(LOGS_DIR)