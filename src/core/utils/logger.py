import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import logging
import os
import yaml
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器"""
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 创建文件处理器（带轮转）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(module_name):
    """获取指定模块的日志记录器"""
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             'config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确定日志文件路径
    log_config = config['logging']
    if module_name in log_config['file_handlers']:
        log_file = log_config['file_handlers'][module_name]
    else:
        log_file = os.path.join(config['paths']['logs'], f"{module_name}.log")
    
    # 设置日志记录器
    return setup_logger(module_name, log_file, getattr(logging, log_config['level'])) 