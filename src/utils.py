import yaml
import logging
import os
import csv
from datetime import datetime

def load_yaml(filepath):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def setup_logger(name, log_dir="outputs/logs"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
        
        c_format = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
    return logger

def save_csv(data, filepath, headers=None):
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists and headers:
            writer.writerow(headers)
        if isinstance(data, list) and len(data)>0 and isinstance(data[0], list):
            writer.writerows(data)
        else:
            writer.writerow(data)
