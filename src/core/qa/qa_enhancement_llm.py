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
        logging.FileHandler("qa_enhancement.log"),
        logging.StreamHandler()
    ]
)

def extract_json_from_text(text):
    """从文本中提取JSON部分"""
    # 尝试提取完整的JSON数组
    json_array_pattern = r'(\[\s*\{.*\}\s*\])'
    json_array_matches = re.findall(json_array_pattern, text, re.DOTALL)
    
    if json_array_matches:
        # 尝试解析提取出的每个JSON数组
        for json_str in json_array_matches:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    # 如果上面的方法失败，尝试提取单个JSON对象并构建数组
    json_obj_pattern = r'(\{[\s\S]*?\})'
    json_obj_matches = re.findall(json_obj_pattern, text)
    
    if json_obj_matches:
        qa_pairs = []
        for json_str in json_obj_matches:
            try:
                obj = json.loads(json_str)
                if "question" in obj and "answer" in obj:
                    qa_pairs.append(obj)
            except json.JSONDecodeError:
                continue
        
        if qa_pairs:
            return qa_pairs
    
    # 如果上述方法都失败，尝试通过规则提取问题和答案对
    question_pattern = r'"question"\s*:\s*"([^"]*)"'
    answer_pattern = r'"answer"\s*:\s*"([^"]*)"'
    
    questions = re.findall(question_pattern, text)
    answers = re.findall(answer_pattern, text)
    
    if questions and answers and len(questions) == len(answers):
        qa_pairs = []
        for i in range(len(questions)):
            qa_pairs.append({
                "question": questions[i],
                "answer": answers[i]
            })
        return qa_pairs
    
    return None

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='使用LLM增强问答对质量')
    parser.add_argument('--input', required=True, help='问答对JSON文件路径')
    parser.add_argument('--output', default='enhanced_qa_pairs.json', help='输出文件路径')
    parser.add_argument('--config', default='models_config.yaml', help='模型配置文件')
    parser.add_argument('--batch_size', type=int, default=5, help='批处理问答对数量')
    parser.add_argument('--enhance_type', type=str, default='all', choices=['all', 'paraphrase', 'style', 'perturb', 'expand', 'shorten'], help='增强类型')
    parser.add_argument('--num_variants', type=int, default=3, help='每个问答对生成变体数量')
    args = parser.parse_args()
    
    # 初始化模型调用器
    model_caller = ModelCaller(config_path=args.config)
    
    # 读取问答对
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            content = f.read()
            # 处理文件可能开头的BOM标记或特殊字符
            if content.startswith('\ufeff') or content.startswith(' '):
                content = content.lstrip('\ufeff').lstrip()
            qa_data = json.loads(content)
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
            
            # 构建多样化增强prompt
            enhance_instruction = ''
            if args.enhance_type == 'all':
                enhance_instruction = '对每个问答对进行同义改写、风格变换、问题扰动、答案扩写和缩写，生成多样化变体。'
            elif args.enhance_type == 'paraphrase':
                enhance_instruction = '对每个问答对进行同义改写，生成多样化表达。'
            elif args.enhance_type == 'style':
                enhance_instruction = '对每个问答对进行风格变换，如正式/口语化表达。'
            elif args.enhance_type == 'perturb':
                enhance_instruction = '对问题部分进行轻微扰动，如换词、调整语序、增加背景信息。'
            elif args.enhance_type == 'expand':
                enhance_instruction = '对答案进行扩写，增加细节。'
            elif args.enhance_type == 'shorten':
                enhance_instruction = '对答案进行缩写，提炼要点。'
            prompt = f"""
你是一个高级供应链技术专家，需要对以下问答对进行增强：

主题: {topic}

增强要求：
{enhance_instruction}
每个问答对请生成{args.num_variants}个不同变体。

以下是需要增强的问答对JSON数组:
{batch_json}

请直接返回增强后的问答对JSON数组（不要添加任何其他说明或前缀），格式如下：
[
  {{
    "question": "增强后的问题1?",
    "answer": "增强后的答案1..."
  }},
  ... 更多问答对 ...
]
"""
            
            # 调用模型
            try:
                response = model_caller.call_model('qa_enhancement', prompt)
                logging.info(f"批次 {batch_start//args.batch_size + 1} 的问答对增强完成，解析结果...")
                
                # 调试输出
                # logging.debug(f"模型返回的原始内容: {response}")
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
                            logging.warning(f"批次 {batch_start//args.batch_size + 1} 的返回结果格式错误，尝试更深入的提取")
                            extracted_batch = extract_json_from_text(response)
                            if extracted_batch:
                                enhanced_batch = extracted_batch
                                logging.info(f"成功通过正则表达式提取到 {len(enhanced_batch)} 个问答对")
                            else:
                                logging.error(f"无法解析批次 {batch_start//args.batch_size + 1} 的返回结果，使用原问答对")
                                enhanced_qa_pairs.extend(current_batch)
                                continue
                    else:
                        logging.warning(f"批次 {batch_start//args.batch_size + 1} 的返回结果格式错误，尝试更深入的提取")
                        extracted_batch = extract_json_from_text(response)
                        if extracted_batch:
                            enhanced_batch = extracted_batch
                            logging.info(f"成功通过正则表达式提取到 {len(enhanced_batch)} 个问答对")
                        else:
                            logging.error(f"无法解析批次 {batch_start//args.batch_size + 1} 的返回结果，使用原问答对")
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
                logging.warning(f"批次 {batch_start//args.batch_size + 1} 的返回结果不是有效JSON，尝试提取有效部分")
                extracted_batch = extract_json_from_text(response)
                if extracted_batch:
                    enhanced_batch = extracted_batch
                    enhanced_qa_pairs.extend(enhanced_batch)
                    logging.info(f"成功通过正则表达式提取到 {len(enhanced_batch)} 个问答对")
                else:
                    logging.error(f"无法解析批次 {batch_start//args.batch_size + 1} 的返回结果，使用原问答对")
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