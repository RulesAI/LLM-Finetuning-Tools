import logging
import sys
import yaml
import anthropic
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

def test_anthropic_models():
    """获取可用的Claude模型列表"""
    try:
        # 从配置文件读取API密钥
        with open('models_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        api_key = config['api_keys']['anthropic']
        logging.info("尝试获取可用的Claude模型列表...")
        
        # 使用requests直接调用API
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        response = requests.get(
            "https://api.anthropic.com/v1/models",
            headers=headers
        )
        
        if response.status_code == 200:
            models = response.json()
            logging.info(f"可用模型列表: {json.dumps(models, indent=2)}")
            return models['data']
        else:
            logging.error(f"获取模型列表失败: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        logging.error(f"获取模型列表失败: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return []

def test_anthropic_api():
    """测试Anthropic (Claude) API连接"""
    try:
        # 从配置文件读取API密钥
        with open('models_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        api_key = config['api_keys']['anthropic']
        logging.info("成功读取Anthropic API密钥")
        
        # 输出API密钥的前10个和后10个字符，中间用星号替代
        masked_key = api_key[:10] + "*" * (len(api_key) - 20) + api_key[-10:]
        logging.info(f"使用的API密钥: {masked_key}")
        
        # 初始化Anthropic客户端
        client = anthropic.Anthropic(api_key=api_key)
        logging.info("成功初始化Anthropic客户端")
        
        # 获取可用模型列表
        available_models = test_anthropic_models()
        model_ids = [model['id'] for model in available_models] if available_models else []
        
        # 如果没有获取到模型列表，使用固定的最新模型列表
        if not model_ids:
            model_ids = [
                "claude-3-7-sonnet-20250219",
                "claude-3-5-sonnet-20241022", 
                "claude-3-5-haiku-20241022",
                "claude-3-5-sonnet-20240620", 
                "claude-3-haiku-20240307",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229"
            ]
        
        # 尝试直接使用消息API
        logging.info("尝试直接使用消息API...")
        for model_id in model_ids:
            logging.info(f"尝试使用模型: {model_id}")
            
            headers = {
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "x-api-key": api_key
            }
            
            data = {
                "model": model_id,
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "请回复'Anthropic API连接测试成功'"}
                ]
            }
            
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logging.info(f"Claude API返回结果: {result.get('content', [{'text': '无内容'}])[0].get('text')}")
                    return True
                else:
                    logging.warning(f"使用模型 {model_id} 失败: {response.status_code} - {response.text}")
            except Exception as e:
                logging.warning(f"使用模型 {model_id} 请求出错: {str(e)}")
                continue
        
        logging.error("所有模型测试均失败")
        
        # 尝试检查账户状态
        try:
            account_headers = {
                "x-api-key": api_key,
                "content-type": "application/json"
            }
            
            account_response = requests.get(
                "https://api.anthropic.com/v1/account",
                headers=account_headers
            )
            
            logging.info(f"账户状态: {account_response.status_code}")
            if account_response.status_code == 200:
                logging.info(f"账户信息: {account_response.json()}")
            else:
                logging.warning(f"获取账户信息失败: {account_response.status_code} - {account_response.text}")
        except Exception as e:
            logging.warning(f"检查账户状态失败: {str(e)}")
        
        return False
        
    except Exception as e:
        logging.error(f"Claude API连接测试失败: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logging.info("开始测试Claude API连接...")
    
    # 测试API连接
    api_result = test_anthropic_api()
    
    if api_result:
        logging.info("✅ Claude API连接测试成功")
        sys.exit(0)
    else:
        logging.error("❌ Claude API连接测试失败")
        sys.exit(1) 