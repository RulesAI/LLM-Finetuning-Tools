import logging
import sys
from model_utils import ModelCaller

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def test_openai_connection():
    """测试OpenAI API连接"""
    try:
        caller = ModelCaller()
        response = caller.call_model('qa_generation', '你好，这是一个API连接测试。')
        logging.info(f"OpenAI API连接测试成功，回复: {response[:50]}...")
        return True
    except Exception as e:
        logging.error(f"OpenAI API连接测试失败: {str(e)}")
        return False

def test_anthropic_connection():
    """测试Anthropic API连接"""
    try:
        caller = ModelCaller()
        response = caller.call_model('document_processing', '你好，这是一个API连接测试。')
        logging.info(f"Anthropic API连接测试成功，回复: {response[:50]}...")
        return True
    except Exception as e:
        logging.error(f"Anthropic API连接测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    logging.info("开始API连接测试...")
    
    openai_success = test_openai_connection()
    anthropic_success = test_anthropic_connection()
    
    if openai_success and anthropic_success:
        logging.info("✅ 所有API连接测试均成功")
        return 0
    elif openai_success:
        logging.warning("⚠️ 仅OpenAI API连接测试成功，Anthropic API连接测试失败")
        return 1
    elif anthropic_success:
        logging.warning("⚠️ 仅Anthropic API连接测试成功，OpenAI API连接测试失败")
        return 1
    else:
        logging.error("❌ 所有API连接测试均失败")
        return 2

if __name__ == "__main__":
    sys.exit(main()) 