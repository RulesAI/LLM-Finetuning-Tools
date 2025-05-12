#!/usr/bin/env python3
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
import re
import logging
import argparse
from src.core.utils.model_utils import ModelCaller

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/fix_json.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fix_json")

def fix_json_string(json_str, model_caller=None):
    """
    使用多种方法尝试修复JSON字符串
    
    参数:
    json_str (str): 需要修复的JSON字符串
    model_caller (ModelCaller, 可选): 用于调用LLM进行修复的ModelCaller实例
    
    返回:
    dict or list: 修复后的JSON数据
    """
    # 首先尝试直接解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.info(f"JSON解析失败: {str(e)}")
    
    # 尝试简单修复
    try:
        # 修复常见问题：额外的逗号, 缺少引号等
        fixed_str = re.sub(r',\s*}', '}', json_str)
        fixed_str = re.sub(r',\s*]', ']', fixed_str)
        
        # 处理缺少引号的键
        fixed_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', fixed_str)
        
        return json.loads(fixed_str)
    except json.JSONDecodeError:
        logger.info("简单修复失败")
    
    # 尝试提取JSON数组或对象
    try:
        # 尝试查找和提取完整的JSON对象或数组
        matches = []
        # 寻找JSON对象
        obj_matches = re.finditer(r'({[^{}]*(?:{[^{}]*}[^{}]*)*})', json_str)
        for match in obj_matches:
            try:
                obj = json.loads(match.group(0))
                matches.append(obj)
                logger.info(f"成功提取JSON对象: {match.group(0)[:50]}...")
            except:
                pass
        
        # 寻找JSON数组
        array_match = re.search(r'(\[\s*{.*}\s*\])', json_str, re.DOTALL)
        if array_match:
            try:
                array = json.loads(array_match.group(0))
                return array
            except:
                logger.info("JSON数组提取失败")
        
        if matches:
            return matches[0] if len(matches) == 1 else matches
    except Exception as e:
        logger.error(f"正则提取失败: {str(e)}")
    
    # 尝试使用LLM修复
    if model_caller:
        try:
            prompt = f"""
请修复以下无效的JSON字符串，只返回修复后的有效JSON，不要包含任何说明：

```
{json_str[:2000]}  # 限制长度防止过长
```

请注意：
1. 只返回修复后的JSON，不要包含其他说明
2. 如果内容太长无法完全修复，请尝试修复可见部分并确保语法正确
"""
            logger.info("尝试使用LLM修复...")
            response = model_caller.call_model("qa_enhancement", prompt)
            
            # 从响应中提取JSON
            json_match = re.search(r'({.*}|\[.*\])', response, re.DOTALL)
            if json_match:
                fixed_json = json_match.group(0)
                try:
                    return json.loads(fixed_json)
                except:
                    logger.error("LLM修复仍然无法解析")
            else:
                logger.error("LLM响应中未找到JSON")
        except Exception as e:
            logger.error(f"LLM修复失败: {str(e)}")
    
    # 所有方法都失败，尝试构建新的JSON结构
    logger.warning("所有自动修复方法失败，尝试构建新的结构")
    try:
        # 识别键值对并构建新的结构
        data = []
        segments = re.split(r'\{|\}', json_str)
        for segment in segments:
            if not segment.strip():
                continue
                
            obj = {}
            # 查找所有键值对
            pairs = re.findall(r'"([^"]+)"\s*:\s*("[^"]*"|[0-9]+|true|false|null|\[[^\]]*\])', segment)
            for key, value in pairs:
                try:
                    obj[key] = json.loads(value)
                except:
                    obj[key] = value.strip('"')
            
            if obj:
                data.append(obj)
        
        if data:
            return data
    except Exception as e:
        logger.error(f"构建新结构失败: {str(e)}")
    
    # 最后的手段：返回空结构
    logger.error("所有修复方法都失败")
    return []

def fix_json_file(input_file, output_file, config_path=None):
    """
    修复JSON文件
    
    参数:
    input_file (str): 输入文件路径
    output_file (str): 输出文件路径
    config_path (str, 可选): 模型配置文件路径
    
    返回:
    bool: 是否成功修复
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 读取文件内容
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 初始化ModelCaller
        model_caller = None
        if config_path:
            try:
                model_caller = ModelCaller(config_path)
                logger.info(f"已初始化ModelCaller，使用配置: {config_path}")
            except Exception as e:
                logger.error(f"初始化ModelCaller失败: {str(e)}")
        
        # 尝试修复JSON
        fixed_data = fix_json_string(content, model_caller)
        
        # 将修复后的数据写回文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fixed_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已将修复后的JSON保存到 {output_file}")
        return True
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='修复无效的JSON文件')
    parser.add_argument('--input', required=True, help='输入JSON文件路径')
    parser.add_argument('--output', required=True, help='输出JSON文件路径')
    parser.add_argument('--config', default='src/config/models_config.yaml', help='模型配置文件路径')
    
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    # 修复JSON文件
    success = fix_json_file(args.input, args.output, args.config)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 