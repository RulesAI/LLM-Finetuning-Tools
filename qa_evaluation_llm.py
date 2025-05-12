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
        logging.FileHandler("quality_check.log"),
        logging.StreamHandler()
    ]
)

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='使用LLM评估问答对质量')
    parser.add_argument('--input', required=True, help='问答对JSON文件路径')
    parser.add_argument('--output', default='high_quality_qa.json', help='输出文件路径')
    parser.add_argument('--config', default='models_config.yaml', help='模型配置文件')
    parser.add_argument('--min_score', type=int, default=80, help='最低质量分数')
    parser.add_argument('--batch_size', type=int, default=3, help='批处理问答对数量')
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
    
    high_quality_qa_data = []
    all_evaluation_results = []
    
    # 对每组问答对进行评估
    for group_idx, qa_group in enumerate(qa_data):
        logging.info(f"正在评估主题组 {group_idx+1}/{len(qa_data)}: {qa_group.get('topic', '未知主题')}")
        
        topic = qa_group.get('topic', '供应链技术')
        qa_pairs = qa_group.get('qa_pairs', [])
        segment_id = qa_group.get('segment_id', group_idx)
        
        high_quality_pairs = []
        group_evaluation_results = []
        
        # 批量处理问答对
        for batch_start in range(0, len(qa_pairs), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(qa_pairs))
            current_batch = qa_pairs[batch_start:batch_end]
            
            if not current_batch:
                continue
                
            logging.info(f"评估批次 {batch_start//args.batch_size + 1}, 包含 {len(current_batch)} 个问答对")
            
            # 依次评估批次中的每个问答对
            for qa_idx, qa_pair in enumerate(current_batch):
                question = qa_pair.get('question', '')
                answer = qa_pair.get('answer', '')
                difficulty = qa_pair.get('difficulty', '')
                dimension = qa_pair.get('dimension', '')
                
                if not question or not answer:
                    logging.warning(f"问答对 {qa_idx+1} 缺少问题或答案，跳过评估")
                    continue
                
                # 构建提示词
                prompt = f"""
你是一个供应链技术教育内容质量评估专家，请对以下问答对的质量进行评估:

主题: {topic}
问题: {question}
答案: {answer}
难度级别: {difficulty}
问题维度: {dimension}

请根据以下标准进行评分(0-100分):
1. 问题质量(25分):
   - 问题是否清晰、明确
   - 问题是否与主题相关
   - 问题表述是否自然、符合专业提问方式
   - 问题长度是否合适(不过长也不过短)

2. 答案质量(25分):
   - 答案是否准确、全面
   - 答案是否有足够的专业深度
   - 答案表述是否清晰、连贯
   - 答案长度是否合适(300-600字之间)

3. 问答匹配度(30分):
   - 答案是否直接回答了问题
   - 问题中的关键概念在答案中是否得到充分解释
   - 答案内容与问题要求的维度是否匹配
   - 答案难度是否与问题标注的难度级别一致

4. 教育价值(20分):
   - 问答对是否有助于学习供应链技术
   - 答案是否包含有价值的见解或深入解释
   - 内容是否适合用于模型微调训练

请直接返回以下JSON格式（不要添加任何其他说明）:
{
  "question_score": 问题得分,
  "answer_score": 答案得分,
  "relevance_score": 问答匹配度得分,
  "education_value": 教育价值得分,
  "total_score": 总分,
  "is_high_quality": 总分是否>=80 (true/false),
  "comments": "简短评价和建议"
}
"""
                
                # 调用模型
                try:
                    response = model_caller.call_model('relevance_evaluation', prompt)
                    logging.info(f"问答对 '{question[:20]}...' 评估完成，解析结果...")
                except Exception as e:
                    logging.error(f"调用模型评估问答对出错: {str(e)}")
                    continue
                
                # 尝试解析返回的JSON
                try:
                    # 尝试直接解析
                    evaluation = json.loads(response)
                    
                    # 基本验证
                    required_fields = ["total_score", "is_high_quality"]
                    if not all(field in evaluation for field in required_fields):
                        # 尝试从文本中提取JSON对象
                        json_match = re.search(r'\{\s*".*"\s*:\s*.*\}', response, re.DOTALL)
                        if json_match:
                            try:
                                evaluation = json.loads(json_match.group(0))
                                if not all(field in evaluation for field in required_fields):
                                    # 仍然缺少必要字段，创建默认评估结果
                                    logging.warning(f"评估结果缺少必要字段，使用默认值")
                                    evaluation = {
                                        "question_score": 0,
                                        "answer_score": 0,
                                        "relevance_score": 0,
                                        "education_value": 0,
                                        "total_score": 0,
                                        "is_high_quality": False,
                                        "comments": "评估失败，无法解析结果"
                                    }
                            except:
                                logging.error(f"从文本中提取JSON后解析仍然失败")
                                continue
                        else:
                            logging.error(f"无法在模型返回结果中找到JSON对象")
                            continue
                    
                    # 记录评估结果
                    evaluation_result = {
                        "question": question,
                        "answer": answer[:100] + "...",  # 只存储答案的一部分以节省空间
                        "difficulty": difficulty,
                        "dimension": dimension,
                        "evaluation": evaluation
                    }
                    
                    group_evaluation_results.append(evaluation_result)
                    
                    # 检查是否为高质量问答对
                    is_high_quality = evaluation.get("is_high_quality", False)
                    total_score = evaluation.get("total_score", 0)
                    
                    if is_high_quality or total_score >= args.min_score:
                        high_quality_pairs.append(qa_pair)
                        logging.info(f"问答对 '{question[:20]}...' 被评为高质量，分数: {total_score}")
                    else:
                        logging.info(f"问答对 '{question[:20]}...' 未达标，分数: {total_score}")
                    
                except json.JSONDecodeError:
                    logging.error(f"评估结果解析失败，跳过该问答对")
                    continue
        
        # 如果有高质量问答对，创建高质量问答组
        if high_quality_pairs:
            high_quality_qa_group = {
                "topic": topic,
                "segment_id": segment_id,
                "qa_pairs": high_quality_pairs
            }
            
            high_quality_qa_data.append(high_quality_qa_group)
            
            logging.info(f"主题 '{topic}' 共有 {len(high_quality_pairs)}/{len(qa_pairs)} 个高质量问答对")
        else:
            logging.warning(f"主题 '{topic}' 没有高质量问答对")
        
        # 添加这个组的评估结果
        all_evaluation_results.append({
            "topic": topic,
            "segment_id": segment_id,
            "total_pairs": len(qa_pairs),
            "high_quality_pairs": len(high_quality_pairs),
            "evaluations": group_evaluation_results
        })
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存高质量问答对
    if high_quality_qa_data:
        try:
            # 保存JSON格式
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(high_quality_qa_data, f, ensure_ascii=False, indent=2)
            
            # 保存JSONL格式（ChatGLM格式）
            jsonl_output = args.output.replace('.json', '.jsonl')
            with open(jsonl_output, 'w', encoding='utf-8') as f:
                for group in high_quality_qa_data:
                    for qa in group.get('qa_pairs', []):
                        chatglm_item = {
                            "instruction": f"结合材料，回答下面的问题: {qa.get('question', '')}",
                            "input": "",
                            "output": qa.get('answer', '')
                        }
                        f.write(json.dumps(chatglm_item, ensure_ascii=False) + '\n')
            
            # 统计问答对总数
            total_qa_pairs = sum(len(group.get("qa_pairs", [])) for group in high_quality_qa_data)
            logging.info(f"质量评估完成，共{len(high_quality_qa_data)}个主题，{total_qa_pairs}个高质量问答对")
            logging.info(f"结果已保存到: {args.output} 和 {jsonl_output}")
            
            # 保存评估结果
            evaluation_output = args.output.replace('.json', '_evaluation.json')
            with open(evaluation_output, 'w', encoding='utf-8') as f:
                json.dump(all_evaluation_results, f, ensure_ascii=False, indent=2)
            
            logging.info(f"评估详情已保存到: {evaluation_output}")
            
            return True
            
        except Exception as e:
            logging.error(f"保存结果出错: {str(e)}")
            return False
    else:
        logging.error("没有找到任何高质量问答对")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        logging.warning("质量评估过程出现错误，请检查日志获取详细信息") 