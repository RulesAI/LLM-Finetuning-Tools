import yaml
import os
import logging
from tenacity import retry, wait_random_exponential, stop_after_attempt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class ModelCaller:
    def __init__(self, config_path="models_config.yaml"):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化API客户端
        self._init_clients()
    
    def _init_clients(self):
        """初始化API客户端"""
        # 初始化OpenAI客户端
        try:
            import openai
            openai.api_key = self.config['api_keys']['openai']
            self.openai_client = openai
            logging.info("已初始化OpenAI客户端")
        except (ImportError, KeyError) as e:
            logging.warning(f"OpenAI客户端初始化失败: {str(e)}")
            self.openai_client = None
        
        # 初始化Anthropic客户端
        try:
            import anthropic
            self.anthropic_client = anthropic.Anthropic(api_key=self.config['api_keys']['anthropic'])
            logging.info("已初始化Anthropic客户端")
        except (ImportError, KeyError) as e:
            logging.warning(f"Anthropic客户端初始化失败: {str(e)}")
            self.anthropic_client = None
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def call_model(self, task_type, prompt):
        """调用指定任务类型的模型"""
        model_name = self.config['models'][task_type]
        logging.info(f"使用模型 {model_name} 处理任务: {task_type}")
        
        # 根据模型名称决定调用哪个API
        if model_name.startswith('gpt'):
            if not self.openai_client:
                raise ValueError("OpenAI客户端未初始化")
            return self._call_openai(model_name, prompt)
        elif model_name.startswith('claude'):
            if not self.anthropic_client:
                raise ValueError("Anthropic客户端未初始化")
            return self._call_anthropic(model_name, prompt)
        # 添加其他模型API的支持...
        else:
            raise ValueError(f"不支持的模型: {model_name}")
    
    def _call_openai(self, model, prompt):
        """调用OpenAI API"""
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config['model_parameters']['temperature'],
                max_tokens=self.config['model_parameters']['max_tokens']
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API调用失败: {str(e)}")
            raise e
    
    def _call_anthropic(self, model, prompt):
        """调用Anthropic API"""
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=self.config['model_parameters']['max_tokens'],
                temperature=self.config['model_parameters']['temperature'],
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Anthropic API调用失败: {str(e)}")
            raise e 