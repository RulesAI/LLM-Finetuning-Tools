import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import argparse
import json
import logging
import os
import re
from src.core.utils.model_utils import ModelCaller

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("semantic_segmentation.log"),
        logging.StreamHandler()
    ]
)

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='使用LLM进行语义分段')
    parser.add_argument('--input', required=True, help='输入文件路径')
    parser.add_argument('--output', default='segmented_content.json', help='输出文件路径')
    parser.add_argument('--config', default='models_config.yaml', help='模型配置文件')
    parser.add_argument('--chunk_size', type=int, default=800, help='目标段落字符数')
    parser.add_argument('--overlap', type=int, default=150, help='段落重叠字符数')
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
你是一个智能文档处理专家，需要将以下供应链技术文档按照语义分段，使每个段落保持完整的语义和主题。

要求:
1. 将文档分成大约{args.chunk_size}字符的段落，允许{args.overlap}字符的重叠
2. 确保每个段落具有完整的语义，不要在句子中间或主题讨论中间切分
3. 段落应该尽可能保持主题的一致性和完整性
4. 输出为JSON格式，每个段落包含segment_id和content两个字段

请直接输出以下格式的JSON数组（不需要任何前言和说明）:
[
  {{"segment_id": 0, "content": "第一个段落内容..."}},
  {{"segment_id": 1, "content": "第二个段落内容..."}},
  ...
]

文档内容:
{content}
"""
    
    # 调用模型
    logging.info("正在使用LLM进行语义分段...")
    try:
        segmented_result = model_caller.call_model('semantic_segmentation', prompt)
        logging.info("语义分段完成，正在解析结果")
    except Exception as e:
        logging.error(f"调用模型进行语义分段出错: {str(e)}")
        return False
    
    # 尝试解析返回的JSON
    try:
        # 尝试直接解析
        segments = json.loads(segmented_result)
        logging.info(f"成功解析分段结果，共{len(segments)}个段落")
    except json.JSONDecodeError:
        # 如果直接解析失败，尝试在结果中查找JSON数组
        logging.warning("直接解析JSON失败，尝试提取JSON部分")
        json_match = re.search(r'\[\s*\{.*\}\s*\]', segmented_result, re.DOTALL)
        if json_match:
            try:
                segments = json.loads(json_match.group(0))
                logging.info(f"成功从文本中提取并解析JSON，共{len(segments)}个段落")
            except json.JSONDecodeError as e:
                logging.error(f"从文本中提取JSON后解析仍然失败: {str(e)}")
                return False
        else:
            logging.error("无法在模型返回结果中找到JSON数组")
            return False
    
    # 验证结果
    if not isinstance(segments, list) or len(segments) == 0:
        logging.error("解析结果不是有效的段落列表")
        return False
    
    # 补全segment_id（如果缺失）
    for i, segment in enumerate(segments):
        if "segment_id" not in segment:
            segment["segment_id"] = i
        if "content" not in segment:
            logging.warning(f"段落{i}缺少content字段")
            segment["content"] = ""
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存结果
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        logging.info(f"分段结果已保存到: {args.output}")
        return True
    except Exception as e:
        logging.error(f"保存分段结果出错: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        logging.warning("语义分段过程出现错误，请检查日志获取详细信息") 