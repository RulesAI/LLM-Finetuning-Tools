import argparse
import json
import logging
import os
import re
from model_utils import ModelCaller

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("qa_generation.log"),
        logging.StreamHandler()
    ]
)

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='使用LLM生成问答对')
    parser.add_argument('--input', required=True, help='分段内容JSON文件路径')
    parser.add_argument('--output', default='qa_pairs.json', help='输出文件路径')
    parser.add_argument('--config', default='models_config.yaml', help='模型配置文件')
    parser.add_argument('--qa_per_segment', type=int, default=5, help='每个段落生成的问答对数量')
    args = parser.parse_args()
    
    # 初始化模型调用器
    model_caller = ModelCaller(config_path=args.config)
    
    # 读取分段内容
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        logging.info(f"成功读取分段文件: {args.input}, 共{len(segments)}个段落")
    except Exception as e:
        logging.error(f"读取分段文件出错: {str(e)}")
        return False
    
    all_qa_groups = []
    
    # 对每个段落生成问答对
    for i, segment in enumerate(segments):
        logging.info(f"正在处理段落 {i+1}/{len(segments)}...")
        
        # 构建提示词
        prompt = f"""
你是一个供应链技术专家，需要从以下段落中提取主题并生成高质量的问答对。

要求:
1. 首先分析并提取段落的核心技术主题
2. 基于主题和内容，生成{args.qa_per_segment}个不同类型的问答对:
   - 基础概念类问题（定义和解释）
   - 工作原理类问题（如何实现功能）
   - 应用场景类问题（适用于哪些情况）
   - 优缺点对比类问题（与其他技术相比的优势）
   - 挑战和难点类问题（实施中可能遇到的困难）
3. 每个问答对应包含:
   - 问题（清晰、具体，以问号结尾）
   - 答案（全面、准确，300-500字）
   - 难度（basic, medium, advanced, expert, challenge之一）
   - 维度（释义类, 应用类, 对比类, 推导类, 纠错类之一）

段落内容:
{segment.get('content', '')}

请直接返回以下JSON格式（不要添加任何其他说明或前缀）:
{{
  "topic": "提取的主题",
  "qa_pairs": [
    {{
      "question": "问题1?",
      "answer": "答案1...",
      "difficulty": "难度级别",
      "dimension": "问题维度"
    }},
    ... 更多问答对 ...
  ]
}}
"""
        
        # 调用模型
        try:
            response = model_caller.call_model('qa_generation', prompt)
            logging.info(f"段落{i+1}的问答对生成完成，解析结果...")
        except Exception as e:
            logging.error(f"调用模型为段落{i+1}生成问答对出错: {str(e)}")
            continue
        
        # 尝试解析返回的JSON
        try:
            # 尝试直接解析
            qa_group = json.loads(response)
            
            # 基本验证
            if "topic" not in qa_group or "qa_pairs" not in qa_group:
                logging.warning(f"段落{i+1}的返回结果缺少必要字段")
                json_match = re.search(r'\{\s*"topic".*\}\s*\}', response, re.DOTALL)
                if json_match:
                    try:
                        qa_group = json.loads(json_match.group(0))
                    except:
                        logging.error(f"段落{i+1}的返回结果格式错误，跳过该段落")
                        continue
                else:
                    logging.error(f"段落{i+1}的返回结果格式错误，跳过该段落")
                    continue
            
            # 添加段落ID
            qa_group["segment_id"] = segment.get("segment_id", i)
            all_qa_groups.append(qa_group)
            
            logging.info(f"段落{i+1}成功生成{len(qa_group.get('qa_pairs', []))}个问答对，主题: {qa_group.get('topic', '未知')}")
            
        except json.JSONDecodeError:
            logging.error(f"段落{i+1}的返回结果不是有效JSON，跳过该段落")
            continue
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存结果
    if all_qa_groups:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(all_qa_groups, f, ensure_ascii=False, indent=2)
            
            # 统计问答对总数
            total_qa_pairs = sum(len(group.get("qa_pairs", [])) for group in all_qa_groups)
            logging.info(f"问答生成完成，共{len(all_qa_groups)}个主题，{total_qa_pairs}个问答对")
            logging.info(f"结果已保存到: {args.output}")
            return True
            
        except Exception as e:
            logging.error(f"保存问答结果出错: {str(e)}")
            return False
    else:
        logging.error("没有成功生成任何问答对")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        logging.warning("问答生成过程出现错误，请检查日志获取详细信息") 