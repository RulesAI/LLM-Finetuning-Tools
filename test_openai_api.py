import logging
import sys
import yaml
import requests
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def test_openai_models():
    """获取可用的OpenAI模型列表"""
    try:
        # 从配置文件读取API密钥
        with open('models_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        api_key = config['api_keys']['openai']
        logging.info("尝试获取可用的OpenAI模型列表...")
        
        # 使用requests直接调用API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers
        )
        
        if response.status_code == 200:
            models = response.json()
            # 只展示部分GPT模型
            gpt_models = [model for model in models['data'] if 'gpt' in model['id'].lower()][:10]
            logging.info(f"部分GPT模型列表: {json.dumps(gpt_models, indent=2)}")
            return models['data']
        else:
            logging.error(f"获取模型列表失败: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        logging.error(f"获取模型列表失败: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return []

def test_openai_api():
    """测试OpenAI API连接"""
    try:
        # 从配置文件读取API密钥
        with open('models_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        api_key = config['api_keys']['openai']
        
        # 如果API密钥是占位符，提示用户更新
        if api_key == "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx":
            logging.error("API密钥是占位符，请更新models_config.yaml中的OpenAI API密钥")
            return False
            
        logging.info("成功读取OpenAI API密钥")
        
        # 输出API密钥的前10个和后5个字符，中间用星号替代
        if len(api_key) > 15:
            masked_key = api_key[:5] + "*" * (len(api_key) - 10) + api_key[-5:]
            logging.info(f"使用的API密钥: {masked_key}")
        
        # 测试模型
        models_to_try = ["gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"]
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        for model in models_to_try:
            logging.info(f"尝试使用模型: {model}")
            
            data = {
                "model": model,
                "messages": [{"role": "user", "content": "请回复'OpenAI API连接测试成功'"}],
                "max_tokens": 100
            }
            
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    message = result['choices'][0]['message']['content']
                    logging.info(f"OpenAI API返回结果: {message}")
                    return True
                else:
                    logging.warning(f"使用模型 {model} 失败: {response.status_code} - {response.text}")
            except Exception as e:
                logging.warning(f"使用模型 {model} 请求出错: {str(e)}")
                continue
        
        logging.error("所有模型测试均失败")
        return False
        
    except Exception as e:
        logging.error(f"OpenAI API连接测试失败: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logging.info("开始测试OpenAI API连接...")
    
    # 测试API连接
    api_result = test_openai_api()
    
    # 如果API连接测试成功，尝试获取模型列表
    if api_result:
        test_openai_models()
        logging.info("✅ OpenAI API连接测试成功")
        sys.exit(0)
    else:
        logging.error("❌ OpenAI API连接测试失败")
        sys.exit(1) 