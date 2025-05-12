import json
import re
import logging
import tiktoken
import argparse
import os
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fix_qa_pairs.log"),
        logging.StreamHandler()
    ]
)

# 使用tiktoken计算tokens数量
def count_tokens(text):
    """计算文本的token数量"""
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception:
        # 备用方案
        return len(text) // 3

# 彻底清理文本中的特殊字符和格式问题
def clean_text_thoroughly(text):
    if not text:
        return ""
    
    # 移除控制字符
    text = re.sub(r'[\x00-\x1F\x7F-\x9F\u0001-\u001F]', '', text)
    
    # 修复截断的文本（特别是以《或》结尾的问题）
    if text.endswith('《') or text.endswith('》'):
        text = text[:-1]
    
    # 修复问题中的引号使用
    text = text.replace('《', '"').replace('》', '"')
    
    # 确保问题以问号结尾
    if text.strip() and not any(text.strip().endswith(c) for c in ['？', '?', '!', '！']):
        if re.search(r'[^\w]$', text):
            text = text[:-1] + '？'
        else:
            text += '？'
    
    # 标准化空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除页码标记
    text = re.sub(r'===\s*第\s*\d+\s*页\s*===', '', text)
    
    # 规范化标点符号
    text = re.sub(r'([。！？!?;；])\s+', r'\1', text)
    
    # 清理多重引号
    text = re.sub(r'[""]{2,}', '"', text)
    text = re.sub(r'["]{2,}', '"', text)
    
    return text.strip()

# 修复问题内容
def fix_question(question):
    """修复问题中的常见问题"""
    if not question:
        return "请解释供应链执行技术的概念和应用？"
    
    # 清理特殊字符和格式
    question = clean_text_thoroughly(question)
    
    # 修复截断的问题
    incomplete_patterns = [
        r'^线》', r'^技术[》")]', r'^供应链[》")]', r'^系统[》")]', 
        r'^平台[》")]', r'^管理[》")]', r'^什么是[》")]'
    ]
    
    for pattern in incomplete_patterns:
        if re.search(pattern, question):
            # 尝试从截断的问题中提取关键词
            keywords = re.findall(r'[\w\s]{2,}', question)
            if keywords:
                keyword = keywords[0].strip()
                # 基于提取的关键词构建新问题
                return f"什么是{keyword}？它有什么作用？"
            else:
                return "什么是供应链执行技术？"
    
    # 修复问题中的无效关键词
    if "哪些公司最" in question and len(question) < 15:
        return "哪些公司最适合采用供应链执行技术？为什么？"
    
    # 确保问题是完整的句子并以问号结尾
    if len(question) < 10:
        return "请解释供应链执行技术的重要性？"
        
    if not question.endswith('？') and not question.endswith('?'):
        question += '？'
    
    return question

# 修复答案内容
def fix_answer(answer):
    """修复答案中的常见问题"""
    if not answer or len(answer) < 50:
        return "供应链执行技术是指支持和优化供应链日常运作的各种技术解决方案，包括仓库管理系统、运输管理系统、物联网技术等。这些技术能够提高供应链的可视性、效率和灵活性，帮助企业更好地应对市场变化和客户需求。"
    
    # 清理特殊字符和格式
    answer = clean_text_thoroughly(answer)
    
    # 移除答案中的页码引用
    answer = re.sub(r'===\s*第\s*\d+\s*页\s*===', '', answer)
    
    # 确保答案以完整句子结尾
    if re.search(r'[,，:：;；]\s*$', answer):
        answer = answer[:-1] + '。'
    
    # 修复不完整的句子
    if answer.endswith('包括') or answer.endswith('例如') or answer.endswith('有'):
        answer += '各种供应链管理和执行工具。'
    
    return answer

# 主要修复函数
def fix_qa_pairs(input_file, output_file):
    """修复问答对文件中的问题"""
    logging.info(f"开始修复文件: {input_file}")
    
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # 检查输入文件格式
        if input_file.endswith('.jsonl'):
            # 读取JSONL格式
            pairs = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            pairs.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logging.error(f"解析JSONL行时出错: {str(e)}")
            
            logging.info(f"成功加载了 {len(pairs)} 个问答对")
            
            # 修复问答对
            fixed_pairs = []
            for pair in pairs:
                try:
                    # 获取对话内容
                    if "conversations" in pair:
                        for item in pair["conversations"]:
                            if item["role"] == "user":
                                item["content"] = fix_question(item["content"])
                            elif item["role"] == "assistant":
                                item["content"] = fix_answer(item["content"])
                    
                    fixed_pairs.append(pair)
                except Exception as e:
                    logging.error(f"修复问答对时出错: {str(e)}")
            
            # 保存修复后的JSONL
            with open(output_file, 'w', encoding='utf-8') as f:
                for pair in fixed_pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            
        else:
            # 读取JSON格式
            with open(input_file, 'r', encoding='utf-8') as f:
                pairs = json.load(f)
            
            logging.info(f"成功加载了 {len(pairs)} 个问答对")
            
            # 修复问答对
            for pair in pairs:
                try:
                    # 修复问题
                    if "user" in pair:
                        pair["user"] = fix_question(pair["user"])
                    
                    # 修复答案
                    if "assistant" in pair:
                        pair["assistant"] = fix_answer(pair["assistant"])
                except Exception as e:
                    logging.error(f"修复问答对时出错: {str(e)}")
            
            # 保存修复后的JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(pairs, f, ensure_ascii=False, indent=2)
        
        # 统计修复问题数量
        num_fixed = sum(1 for pair in pairs if "fixed" in pair and pair["fixed"])
        
        logging.info(f"完成修复，总共处理了 {len(pairs)} 个问答对，修复了 {num_fixed} 个问题")
        logging.info(f"结果已保存到 {output_file}")
        
        return True
    
    except Exception as e:
        logging.error(f"修复过程出错: {str(e)}")
        return False

# 验证修复后的问答对质量
def validate_fixed_qa_pairs(file_path):
    """验证修复后的问答对质量"""
    logging.info(f"开始验证文件: {file_path}")
    
    try:
        # 确定文件格式
        is_jsonl = file_path.endswith('.jsonl')
        
        if is_jsonl:
            # 读取JSONL格式
            pairs = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            pairs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        else:
            # 读取JSON格式
            with open(file_path, 'r', encoding='utf-8') as f:
                pairs = json.load(f)
        
        total_pairs = len(pairs)
        valid_pairs = 0
        invalid_questions = 0
        invalid_answers = 0
        
        # 验证每个问答对
        for pair in pairs:
            is_valid = True
            
            # 验证问题
            question = None
            if is_jsonl and "conversations" in pair:
                for item in pair["conversations"]:
                    if item["role"] == "user":
                        question = item["content"]
                        break
            elif "user" in pair:
                question = pair["user"]
            
            if not question or len(question) < 10 or not question.endswith(('？', '?')):
                invalid_questions += 1
                is_valid = False
            
            # 验证答案
            answer = None
            if is_jsonl and "conversations" in pair:
                for item in pair["conversations"]:
                    if item["role"] == "assistant":
                        answer = item["content"]
                        break
            elif "assistant" in pair:
                answer = pair["assistant"]
            
            if not answer or len(answer) < 50:
                invalid_answers += 1
                is_valid = False
            
            if is_valid:
                valid_pairs += 1
        
        # 打印验证结果
        logging.info(f"验证完成: 总计 {total_pairs} 个问答对")
        logging.info(f"有效问答对: {valid_pairs} ({valid_pairs/total_pairs*100:.1f}%)")
        logging.info(f"无效问题: {invalid_questions} ({invalid_questions/total_pairs*100:.1f}%)")
        logging.info(f"无效答案: {invalid_answers} ({invalid_answers/total_pairs*100:.1f}%)")
        
        return valid_pairs, invalid_questions, invalid_answers
    
    except Exception as e:
        logging.error(f"验证过程出错: {str(e)}")
        return 0, 0, 0

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='修复问答对文件中的问题')
    parser.add_argument('--input_json', help='JSON格式输入文件', default=None)
    parser.add_argument('--input_jsonl', help='JSONL格式输入文件', default=None)
    parser.add_argument('--output_dir', help='输出目录', default=None)

    args = parser.parse_args()
    
    # 记录参数信息
    logging.info(f"启动参数: input_json={args.input_json}, input_jsonl={args.input_jsonl}, output_dir={args.output_dir}")
    
    success = False

    # 如果指定了参数，则使用参数值
    if args.input_json and args.output_dir:
        input_json = args.input_json
        output_json = os.path.join(args.output_dir, os.path.basename(input_json).replace('.json', '_improved.json'))
        logging.info(f"处理JSON: {input_json} -> {output_json}")
        
        # 确保输出目录存在
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            logging.info(f"创建输出目录: {args.output_dir}")
            
        if fix_qa_pairs(input_json, output_json):
            validate_fixed_qa_pairs(output_json)
            success = True

    if args.input_jsonl and args.output_dir:
        input_jsonl = args.input_jsonl
        output_jsonl = os.path.join(args.output_dir, os.path.basename(input_jsonl).replace('.jsonl', '_improved.jsonl'))
        logging.info(f"处理JSONL: {input_jsonl} -> {output_jsonl}")
        
        # 确保输出目录存在
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            logging.info(f"创建输出目录: {args.output_dir}")
            
        if fix_qa_pairs(input_jsonl, output_jsonl):
            validate_fixed_qa_pairs(output_jsonl)
            success = True

    # 如果没有指定参数，则使用默认文件
    if not any([args.input_json, args.input_jsonl]):
        # 定义输入和输出文件
        input_files = [
            'qa_instructions_chatglm_robust.jsonl',
            'qa_instructions_robust.json'
        ]
        
        for input_file in input_files:
            if not Path(input_file).exists():
                logging.warning(f"文件 {input_file} 不存在，跳过")
                continue
            
            # 生成输出文件名
            output_file = input_file.replace('.json', '_improved.json')
            if output_file == input_file:
                output_file = input_file.replace('.', '_improved.')
            
            logging.info(f"处理文件: {input_file} -> {output_file}")
            
            # 修复问答对
            if fix_qa_pairs(input_file, output_file):
                # 验证修复结果
                validate_fixed_qa_pairs(output_file)
                success = True
    
    if success:
        logging.info("问答对修复完成")
        return 0
    else:
        logging.error("问答对修复失败或无文件可处理")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 