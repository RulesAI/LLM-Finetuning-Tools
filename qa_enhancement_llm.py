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
        logging.FileHandler("qa_enhancement.log"),
        logging.StreamHandler()
    ]
)

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='使用LLM增强问答对质量')
    parser.add_argument('--input', required=True, help='问答对JSON文件路径')
    parser.add_argument('--output', default='enhanced_qa_pairs.json', help='输出文件路径')
    parser.add_argument('--config', default='models_config.yaml', help='模型配置文件')
    parser.add_argument('--batch_size', type=int, default=5, help='批处理问答对数量')
    args = parser.parse_args()
    
    # 初始化模型调用器
    model_caller = ModelCaller(config_path=args.config)
    
    # 读取问答对
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        logging.info(f"成功读取问答文件: {args.input}")
    except Exception as e:
        logging.error(f"读取问答文件出错: {str(e)}")
        return False
    
    enhanced_qa_data = []
    
    # 对每组问答对进行增强
    for group_idx, qa_group in enumerate(qa_data):
        logging.info(f"正在处理主题组 {group_idx+1}/{len(qa_data)}: {qa_group.get('topic', '未知主题')}")
        
        topic = qa_group.get('topic', '供应链技术')
        qa_pairs = qa_group.get('qa_pairs', [])
        segment_id = qa_group.get('segment_id', group_idx)
        
        enhanced_qa_pairs = []
        
        # 批量处理问答对
        for batch_start in range(0, len(qa_pairs), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(qa_pairs))
            current_batch = qa_pairs[batch_start:batch_end]
            
            if not current_batch:
                continue
                
            logging.info(f"处理批次 {batch_start//args.batch_size + 1}, 包含 {len(current_batch)} 个问答对")
            
            # 将批次格式化为JSON字符串
            batch_json = json.dumps(current_batch, ensure_ascii=False)
            
            # 构建提示词
            prompt = f"""
你是一个高级供应链技术专家，需要增强和修复以下问答对的质量。

主题: {topic}

要求:
1. 修复问题中的表述问题:
   - 确保问题清晰、明确、自然，并以问号结尾
   - 修复不完整或截断的问题表述
   - 确保问题与主题相关

2. 增强回答内容:
   - 确保回答准确、全面、连贯
   - 修复不完整或截断的答案
   - 确保回答直接针对问题
   - 优化专业术语使用，保持一致性
   - 保持回答的教育价值和专业性

3. 保持原有的难度和维度标签

以下是需要增强的问答对JSON数组:
{batch_json}

请直接返回增强后的问答对JSON数组（不要添加任何其他说明或前缀）:
[
  {{
    "question": "增强后的问题1?",
    "answer": "增强后的答案1...",
    "difficulty": "原难度级别",
    "dimension": "原问题维度"
  }},
  ... 更多问答对 ...
]
"""
            
            # 调用模型
            try:
                response = model_caller.call_model('qa_enhancement', prompt)
                logging.info(f"批次 {batch_start//args.batch_size + 1} 的问答对增强完成，解析结果...")
            except Exception as e:
                logging.error(f"调用模型增强批次 {batch_start//args.batch_size + 1} 的问答对出错: {str(e)}")
                # 如果增强失败，保留原问答对
                enhanced_qa_pairs.extend(current_batch)
                continue
            
            # 尝试解析返回的JSON
            try:
                # 尝试直接解析
                enhanced_batch = json.loads(response)
                
                # 验证结果
                if not isinstance(enhanced_batch, list):
                    # 尝试从文本中提取JSON数组
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
                    if json_match:
                        try:
                            enhanced_batch = json.loads(json_match.group(0))
                        except:
                            logging.error(f"批次 {batch_start//args.batch_size + 1} 的返回结果格式错误，使用原问答对")
                            enhanced_qa_pairs.extend(current_batch)
                            continue
                    else:
                        logging.error(f"批次 {batch_start//args.batch_size + 1} 的返回结果格式错误，使用原问答对")
                        enhanced_qa_pairs.extend(current_batch)
                        continue
                
                # 检查数量是否一致，如不一致则补齐
                if len(enhanced_batch) != len(current_batch):
                    logging.warning(f"增强后的问答对数量({len(enhanced_batch)})与原始数量({len(current_batch)})不一致")
                    
                    # 找到缺失的问答对并添加回来
                    if len(enhanced_batch) < len(current_batch):
                        enhanced_questions = [item.get('question', '') for item in enhanced_batch]
                        for orig_qa in current_batch:
                            if orig_qa.get('question', '') not in enhanced_questions:
                                enhanced_batch.append(orig_qa)
                                logging.info(f"添加缺失的问答对: {orig_qa.get('question', '')[:30]}...")
                
                # 添加增强后的问答对
                enhanced_qa_pairs.extend(enhanced_batch)
                
                logging.info(f"批次 {batch_start//args.batch_size + 1} 成功增强 {len(enhanced_batch)} 个问答对")
                
            except json.JSONDecodeError:
                logging.error(f"批次 {batch_start//args.batch_size + 1} 的返回结果不是有效JSON，使用原问答对")
                enhanced_qa_pairs.extend(current_batch)
        
        # 创建增强后的问答组
        enhanced_qa_group = {
            "topic": topic,
            "segment_id": segment_id,
            "qa_pairs": enhanced_qa_pairs
        }
        
        enhanced_qa_data.append(enhanced_qa_group)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存结果
    if enhanced_qa_data:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(enhanced_qa_data, f, ensure_ascii=False, indent=2)
            
            # 统计问答对总数
            total_qa_pairs = sum(len(group.get("qa_pairs", [])) for group in enhanced_qa_data)
            logging.info(f"问答增强完成，共{len(enhanced_qa_data)}个主题，{total_qa_pairs}个问答对")
            logging.info(f"结果已保存到: {args.output}")
            return True
            
        except Exception as e:
            logging.error(f"保存增强结果出错: {str(e)}")
            return False
    else:
        logging.error("没有成功增强任何问答对")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        logging.warning("问答增强过程出现错误，请检查日志获取详细信息") 