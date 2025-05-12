import argparse
import json
import logging
import os
from model_utils import ModelCaller

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_cleaning.log"),
        logging.StreamHandler()
    ]
)

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='使用LLM增强文档清洗')
    parser.add_argument('--input', required=True, help='输入文件路径')
    parser.add_argument('--output', required=True, help='输出文件路径')
    parser.add_argument('--config', default='models_config.yaml', help='模型配置文件')
    args = parser.parse_args()
    
    # 初始化模型调用器
    model_caller = ModelCaller(config_path=args.config)
    
    # 读取输入文件
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            content = f.read()
        logging.info(f"成功读取文件: {args.input}, 大小: {len(content)} 字符")
    except Exception as e:
        logging.error(f"读取输入文件出错: {str(e)}")
        return False
    
    # 构建提示词
    prompt = f"""
你是一个专业的文档清洗专家，擅长处理供应链技术文档。请对以下文档内容进行结构化处理和清洗:

1. 识别并标记文档中的结构元素:
   - 标题(不同层级的标题)
   - 段落
   - 图表及其描述
   - 列表(有序和无序)

2. 统一处理专业术语:
   - 识别供应链领域的专业术语(如WMS, TMS, ERP等)
   - 统一术语的表达方式，如将"仓库管理系统"和"Warehouse Management System"统一为"WMS"
   - 保留首次完整表达，后续使用统一缩写

3. 移除重复和冗余内容:
   - 识别并删除重复的段落或句子
   - 合并表达相同含义的内容

4. 清理不应出现的引用:
   - 移除文本中的引用网址、域名和站点引用
   - 去除参考文献标记

5. 增强图表内容与说明的关联:
   - 将图表标题与说明文本关联
   - 保留图表的上下文信息

6. 技术概念关联性强化:
   - 识别技术定义段落
   - 建立术语与其定义的关联
   - 优化专业术语的上下文表达

请直接输出处理完成的文本内容，保持文档的完整性和连贯性。不需要说明你做了什么，只需要输出处理后的内容。

文档内容:
{content}
"""
    
    # 调用模型
    logging.info("正在使用LLM进行文档清洗...")
    try:
        cleaned_content = model_caller.call_model('document_processing', prompt)
        logging.info(f"文档清洗完成，处理后大小: {len(cleaned_content)} 字符")
    except Exception as e:
        logging.error(f"调用模型进行文档清洗出错: {str(e)}")
        return False
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存结果
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        logging.info(f"清洗结果已保存到: {args.output}")
        return True
    except Exception as e:
        logging.error(f"保存清洗结果出错: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        logging.warning("文档清洗过程出现错误，请检查日志获取详细信息") 