import json
import random
import re
import tiktoken
import logging
import argparse
import os
from pathlib import Path
from collections import Counter

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("qa_generation.log"),
        logging.StreamHandler()
    ]
)

# 读取语义分段结果
def load_segments(file_path='segmented_content.json'):
    try:
        logging.info(f"从文件加载分段数据: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载分段文件失败: {str(e)}")
        return []

# 使用tiktoken计算tokens数量
def count_tokens(text):
    """计算文本的token数量"""
    try:
        encoder = tiktoken.get_encoding("cl100k_base")  # ChatGPT模型使用的编码
        return len(encoder.encode(text))
    except Exception:
        # 如果tiktoken失败，使用简单的字符长度作为回退方案
        return len(text) // 3

# 彻底清理文本中的特殊字符和格式问题
def clean_text_thoroughly(text):
    if not text:
        return ""
    
    # 移除所有非打印字符和控制字符
    text = re.sub(r'[\x00-\x1F\x7F-\x9F\u0001-\u001F]', '', text)
    
    # 移除页码标记
    text = re.sub(r'===\s*第\s*\d+\s*页\s*===', '', text)
    
    # 标准化空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 规范化标点符号
    text = re.sub(r'([。！？!?;；])\s+', r'\1', text)
    
    # 移除过多的空行
    text = re.sub(r'\n{2,}', '\n\n', text)
    
    # 移除行尾空格
    text = re.sub(r' +\n', '\n', text)
    
    return text.strip()

# 定义系统提示
SYSTEM_PROMPT = "你是一个专业的供应链技术领域顾问，擅长解释各种供应链执行技术的概念、应用场景和发展趋势。请基于你的专业知识回答问题。"

# 改进的问题模板
question_templates = {
    # 基础难度 - 定义和解释 (What)
    "basic": [
        "什么是{topic}？",
        "{topic}的基本定义是什么？",
        "请简要解释{topic}的概念。",
        "{topic}代表了什么技术？",
        "{topic}包含哪些核心要素？",
    ],
    
    # 中级难度 - 原理和机制 (How & Why)
    "medium": [
        "{topic}的工作原理是什么？",
        "{topic}如何实现其功能？",
        "为什么{topic}在供应链中很重要？",
        "{topic}解决了什么问题？",
        "{topic}的核心机制是如何运作的？",
    ],
    
    # 高级难度 - 应用和示例 (Where & When)
    "advanced": [
        "{topic}在哪些场景下最适合应用？",
        "什么情况下企业应该考虑实施{topic}？",
        "{topic}在什么行业应用最广泛？为什么？",
        "什么时候是引入{topic}的最佳时机？",
        "{topic}能为哪类企业带来最大价值？",
    ],
    
    # 专家难度 - 评估和推理 (Who & Why not)
    "expert": [
        "{topic}有哪些局限性？",
        "企业在考虑{topic}时需要评估哪些因素？",
        "哪些公司最适合采用{topic}？为什么？",
        "为什么有些企业可能不适合实施{topic}？",
        "{topic}相比其他解决方案有什么优缺点？",
    ],
    
    # 挑战类问题 (5W+2H综合)
    "challenge": [
        "实施{topic}时企业通常会面临哪些挑战？",
        "{topic}在当前市场中存在哪些限制？",
        "为什么{topic}在某些场景下难以落地？",
        "企业如何克服{topic}实施过程中的阻碍？",
        "{topic}的成本效益分析关键点有哪些？",
    ]
}

# 不同思考维度的问题生成模板
thinking_dimensions = {
    "释义类": ["basic", "medium"],  # 解释概念和原理
    "应用类": ["advanced"],         # 应用场景和示例
    "对比类": ["expert"],           # 比较优缺点
    "推导类": ["expert", "challenge"],  # 逻辑推理和分析
    "纠错类": ["challenge"]         # 识别误区和纠正
}

# 已知技术术语列表 - 用于辅助主题识别
known_tech_terms = [
    "生成式AI", "仿人工作机器人", "自动驾驶卡车", "最后一公里配送",
    "室内资产实时定位", "自主数据采集", "仓库资源规划", "仓库劳动力预测",
    "仓储数字孪生", "仓库执行系统", "多代理机器人编排", "仓库仿真建模",
    "实时运输可视化", "供应链协同网络", "供应链融合", "物联网", "区块链",
    "WMS", "供应链执行技术", "供应链可视化", "RTTVP", "WES", "自动化技术"
]

# 改进的主题提取方法，更加稳健
def extract_robust_topic(text):
    """增强版主题提取，确保不返回不完整或带格式标记的主题"""
    if not text:
        return "供应链技术"
    
    # 先彻底清理文本
    cleaned_text = clean_text_thoroughly(text)
    
    # 1. 首先尝试识别是否有段落标题或明确的技术定义
    tech_title_match = re.search(r'([\w\s]{2,20})[（(]([^)）]+)[)）]技术[:：]', cleaned_text)
    if tech_title_match and len(tech_title_match.group(1).strip()) > 2:
        candidate = tech_title_match.group(1).strip() + "技术"
        if validate_topic(candidate):
            return candidate
    
    # 2. 检查是否包含已知技术术语（完整匹配）
    for term in known_tech_terms:
        # 确保是主要讨论对象，而不仅仅是提及
        # 查找格式如"xxx技术定义与原理："或"xxx是指"这样的模式
        pattern = f"{term}([：:](技术定义|原理|是指)|是指|指的是|被定义为)"
        if re.search(pattern, cleaned_text, re.IGNORECASE):
            return term
        # 或者在文本中多次出现的核心技术术语
        if cleaned_text.count(term) >= 3 and len(term) > 4:
            return term
    
    # 3. 检查是否有常见的技术描述引导句
    definition_patterns = [
        r'([\w\s]{4,25})(技术定义与原理|是一种|技术是指)',
        r'([\w\s]{4,25})[:：](技术定义|原理|是指|指的是|包括|涵盖)',
        r'([\w\s]{4,25})是[供物仓运输链智能].*?领域.*?的.*?技术'
    ]
    
    for pattern in definition_patterns:
        match = re.search(pattern, cleaned_text)
        if match and len(match.group(1).strip()) > 3:
            candidate = match.group(1).strip()
            # 如果不以"技术"结尾，则添加
            if not candidate.endswith("技术") and "技术" not in candidate:
                candidate += "技术"
            if validate_topic(candidate):
                return candidate
    
    # 4. 提取前100个字符中的核心技术名词
    first_part = cleaned_text[:200]
    tech_keywords = [
        "系统", "平台", "技术", "方案", "解决方案", "服务", "工具", "框架"
    ]
    
    for keyword in tech_keywords:
        # 匹配"XX系统"或"XX平台"等模式
        pattern = r'([\w]{2,15}' + keyword + r')'
        matches = re.findall(pattern, first_part)
        if matches:
            # 选择最长的匹配，避免过短的技术名称
            matches.sort(key=len, reverse=True)
            for match in matches:
                if validate_topic(match) and len(match) > 4:
                    return match
    
    # 5. 避免将子主题或属性提取为主题
    # 检查是否存在"XXX与YYY"或"XXX和YYY"格式，如果存在，考虑提取主题
    conjunction_match = re.search(r'([\w]{2,15})(与|和)(安全|责任|挑战|限制|问题)', cleaned_text)
    if conjunction_match:
        main_term = conjunction_match.group(1)
        if validate_topic(main_term + "技术"):
            return main_term + "技术"
    
    # 6. 从特定语境中提取隐含主题
    if "自动驾驶" in cleaned_text and ("卡车" in cleaned_text or "物流" in cleaned_text):
        return "自动驾驶卡车技术"
    
    if "责任归属" in cleaned_text and "自动驾驶" in cleaned_text:
        return "自动驾驶技术"
    
    # 7. 从完整上下文中识别主题，避免抽取子问题作为主题
    if "安全与责任" in cleaned_text and "自动驾驶" in cleaned_text:
        return "自动驾驶技术"
    
    # 8. 默认回退方案
    if "仓库" in cleaned_text or "WMS" in cleaned_text:
        return "仓库管理技术"
    elif "物流" in cleaned_text:
        return "物流技术"
    elif "供应链" in cleaned_text:
        return "供应链技术"
    
    # 最终默认值
    return "供应链执行技术"

# 验证主题的有效性
def validate_topic(topic):
    """确保主题是有效且完整的，完善版"""
    if not topic:
        return False
    
    # 排除过短的主题
    if len(topic) < 3:
        return False
    
    # 排除格式标记或截断的主题
    invalid_chars = ["》", "《", "「", "」", "..", "…", "。。"]
    for char in invalid_chars:
        if char in topic:
            return False
    
    # 排除纯数字或纯标点的主题
    if re.match(r'^[\d\s.,;:]+$', topic):
        return False
    
    # 排除常见的非主题词
    non_topics = [
        "技术定义", "当前成熟度", "应用场景", "实施与落地", "未来发展", "领先企业",
        "安全与责任", "责任归属", "限制", "挑战", "问题", "的因素", "前处于", "案正位于"
    ]
    for nt in non_topics:
        if topic == nt or topic.startswith(nt) or topic.endswith(nt):
            return False
    
    # 如果主题过于笼统，考虑添加限定词
    if topic in ["技术", "系统", "平台", "方案"]:
        return False
        
    # 确保主题名称格式合理
    if re.match(r'^的[\w]{1,3}$', topic):  # 以"的X"开头的短词，可能是片段
        return False
    
    return True

# 生成高质量答案
def generate_quality_answer(text, max_tokens=350):
    """生成一个良好格式的高质量答案"""
    if not text:
        return "对不起，我没有足够的信息来回答这个问题。"
    
    # 彻底清理文本
    cleaned_text = clean_text_thoroughly(text)
    
    # 如果清理后的文本已经足够简短，可以直接使用
    if count_tokens(cleaned_text) <= max_tokens:
        return cleaned_text
    
    # 将文本分成句子
    sentences = re.split(r'([。！？!?;；])', cleaned_text)
    proper_sentences = []
    
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            proper_sentences.append(sentences[i] + sentences[i+1])
        else:
            proper_sentences.append(sentences[i])
    
    # 过滤空句子
    proper_sentences = [s.strip() for s in proper_sentences if s.strip()]
    
    # 按重要性选择句子
    important_sentences = []
    
    # 优先包含定义或概念的句子
    definition_patterns = ["是指", "定义", "概念", "原理", "技术", "系统", "指的是"]
    for sentence in proper_sentences:
        for pattern in definition_patterns:
            if pattern in sentence:
                important_sentences.append(sentence)
                break
        if len(important_sentences) >= 1:
            break
    
    # 添加包含关键信息的句子
    key_patterns = ["应用场景", "挑战", "优势", "趋势", "特点", "价值", "案例"]
    for pattern in key_patterns:
        for sentence in proper_sentences:
            if pattern in sentence and sentence not in important_sentences:
                important_sentences.append(sentence)
            if count_tokens(" ".join(important_sentences)) > max_tokens * 0.7:
                break
        if count_tokens(" ".join(important_sentences)) > max_tokens * 0.7:
            break
    
    # 补充更多句子直到接近最大token
    for sentence in proper_sentences:
        if sentence not in important_sentences:
            if count_tokens(" ".join(important_sentences + [sentence])) <= max_tokens:
                important_sentences.append(sentence)
            else:
                break
    
    # 形成最终答案
    answer = " ".join(important_sentences)
    
    # 确保答案不为空，并且没有特殊字符
    if not answer or len(answer) < 50:
        # 如果重要句子提取失败，使用前几个句子
        answer = " ".join(proper_sentences[:3])
    
    return clean_text_thoroughly(answer)

# 改进的问题生成函数，检查主题与段落内容的关联性
def generate_validated_question(segment_content, dimension):
    """生成验证过的问题，确保与段落内容相关"""
    # 提取段落的主题
    topic = extract_robust_topic(segment_content)
    
    # 检查主题提取质量
    if topic.endswith("的因素") or topic.startswith("的") or len(topic) < 4:
        # 使用备选方案
        if "自动驾驶" in segment_content:
            topic = "自动驾驶技术"
        elif "仓库" in segment_content:
            topic = "仓库管理技术"
        elif "供应链" in segment_content:
            topic = "供应链技术"
    
    # 验证主题与段落内容的关联性
    topic_words = re.findall(r'\w+', topic)
    content_related = False
    
    for word in topic_words:
        if len(word) > 1 and word in segment_content:
            content_related = True
            break
    
    if not content_related and len(topic) > 2:
        # 主题与内容似乎不相关，尝试备用方案
        logging.warning(f"提取的主题'{topic}'与段落内容可能不相关，尝试备用方案")
        
        # 检查段落中常见技术关键词
        for tech in known_tech_terms:
            if tech in segment_content:
                topic = tech
                content_related = True
                break
    
    if not content_related:
        # 如果仍然找不到相关主题，使用更一般的主题
        logging.warning(f"无法确定段落相关主题，使用一般主题")
        topic = "供应链执行技术"
    
    # 随机选择一个问题模板
    templates = []
    if dimension in thinking_dimensions:
        difficulty_types = thinking_dimensions[dimension]
        for difficulty in difficulty_types:
            templates.extend(question_templates[difficulty])
    else:
        # 默认使用medium难度
        templates = question_templates["medium"]
    
    template = random.choice(templates)
    
    # 生成问题并确保以问号结尾
    question = template.format(topic=topic)
    if not question.endswith("？"):
        question += "？"
    
    return {
        "question": question,
        "topic": topic,
        "dimension": dimension,
        "difficulty": next((d for d in question_templates if template in question_templates[d]), "medium")
    }

# 生成多样化问题
def generate_diverse_questions(segment, dimensions_to_use=None):
    """为一个段落生成多样的问题集"""
    # 使用默认维度或指定维度
    segment_content = segment.get("content", "")
    if not segment_content:
        return []
    
    # 如果未指定维度，随机选择2-3个不同维度
    if not dimensions_to_use:
        dimensions_to_use = random.sample(list(thinking_dimensions.keys()), k=min(3, len(thinking_dimensions)))
    
    # 生成问题集合
    questions = []
    used_templates = set()  # 跟踪已使用的模板，避免重复
    
    for dimension in dimensions_to_use:
        # 每个维度尝试生成1-2个问题
        count = random.randint(1, 2)
        for _ in range(count):
            for attempt in range(3):  # 最多尝试3次找到不重复模板
                question_data = generate_validated_question(segment_content, dimension)
                question_text = question_data["question"]
                
                # 检查问题是否重复
                if question_text not in used_templates:
                    used_templates.add(question_text)
                    
                    # 添加问题及其元数据
                    questions.append({
                        "system": SYSTEM_PROMPT,
                        "user": question_text,
                        "assistant": generate_quality_answer(segment_content),
                        "topic": question_data["topic"],
                        "dimension": dimension,
                        "difficulty": question_data["difficulty"],
                        "segment_id": segment.get("id")
                    })
                    break
    
    # 检查主题一致性
    topic_counter = Counter([q["topic"] for q in questions])
    
    # 如果有多种主题，选择最常见的作为标准
    if len(topic_counter) > 1:
        main_topic = topic_counter.most_common(1)[0][0]
        for q in questions:
            if q["topic"] != main_topic and "责任归属" in q["topic"]:
                # 特别处理"责任归属"相关问题
                if "自动驾驶" in segment_content:
                    q["topic"] = "自动驾驶技术"
    
    return questions

# 验证所有生成的问答对
def validate_qa_pairs(all_pairs, segments):
    """验证并修复问答对的质量问题"""
    validated_pairs = []
    fixed_count = 0
    
    for pair in all_pairs:
        needs_fixing = False
        
        # 验证问题质量
        question = pair["user"]
        if not question or len(question) < 10 or "》" in question or "《" in question:
            needs_fixing = True
            logging.warning(f"发现问题质量问题: {question}")
        
        # 验证答案质量
        answer = pair["assistant"]
        if not answer or len(answer) < 100 or "===" in answer:
            needs_fixing = True
            logging.warning(f"发现答案质量问题，长度: {len(answer)}")
        
        # 如果需要修复，重新生成问答对
        if needs_fixing:
            try:
                segment_id = pair["segment_id"]
                dimension = pair["dimension"]
                
                # 获取原始段落内容
                segment_content = None
                for seg in segments:
                    if seg["segment_id"] == segment_id:
                        segment_content = seg["content"]
                        break
                
                if segment_content:
                    # 重新生成问题
                    question_data = generate_validated_question(segment_content, dimension)
                    pair["user"] = question_data["question"]
                    pair["topic"] = question_data["topic"]
                    pair["difficulty"] = question_data["difficulty"]
                    
                    # 重新生成答案
                    pair["assistant"] = generate_quality_answer(segment_content)
                    
                    fixed_count += 1
                    logging.info(f"修复了问答对: {question_data['question']}")
            except Exception as e:
                logging.error(f"修复问答对时出错: {str(e)}")
        
        # 再次验证修复后的问答对
        if validate_single_qa_pair(pair):
            validated_pairs.append(pair)
    
    logging.info(f"验证完成: 总共 {len(all_pairs)} 对, 修复了 {fixed_count} 对, 保留了 {len(validated_pairs)} 对")
    return validated_pairs, fixed_count

# 验证单个问答对
def validate_single_qa_pair(pair):
    """验证单个问答对是否合格"""
    # 检查问题
    question = pair["user"]
    if not question or len(question) < 5:
        return False
    
    # 检查答案
    answer = pair["assistant"]
    if not answer or len(answer) < 50:
        return False
    
    # 检查特殊字符
    special_chars = ["》", "《", "\u0001", "==="]
    for char in special_chars:
        if char in question or char in answer:
            return False
    
    return True

# 将结果保存到文件
def save_to_files(validated_pairs, chatglm_conversations, output_dir, stats):
    """保存结果到文件"""
    try:
        # 确保输出目录存在
        output_dir_str = str(output_dir)
        logging.info(f"准备保存文件到目录: {output_dir_str}")
        
        # 使用os.makedirs替代Path.mkdir
        os.makedirs(output_dir_str, exist_ok=True)
        logging.info(f"目录已创建或确认存在: {output_dir_str}")
        
        # 检查目录是否确实存在
        if not os.path.exists(output_dir_str):
            logging.error(f"尝试创建目录后，目录仍然不存在: {output_dir_str}")
            # 尝试其他方法
            os.system(f"mkdir -p {output_dir_str}")
            logging.info(f"通过系统命令创建目录: mkdir -p {output_dir_str}")
        
        # 再次检查目录是否存在
        if not os.path.exists(output_dir_str):
            logging.error(f"无法创建目录: {output_dir_str}")
            # 尝试使用当前目录作为回退
            output_dir_str = "."
            logging.warning(f"将使用当前目录保存文件: {output_dir_str}")
        
        # 保存JSONL文件
        jsonl_path = os.path.join(output_dir_str, 'qa_instructions_chatglm_robust.jsonl')
        logging.info(f"保存JSONL文件到: {jsonl_path}")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for conv in chatglm_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        # 保存JSON文件
        json_path = os.path.join(output_dir_str, 'qa_instructions_robust.json')
        logging.info(f"保存JSON文件到: {json_path}")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(validated_pairs, f, ensure_ascii=False, indent=2)
        
        # 打印统计信息
        logging.info(f"生成了 {len(validated_pairs)} 个Q/A指令对")
        for key, value in stats.items():
            logging.info(f"{key}: {value}")
        logging.info(f"结果已保存到 {jsonl_path} 和 {json_path} 文件")
        
        return True
    except Exception as e:
        logging.error(f"保存问答对文件时出错: {str(e)}")
        # 打印更详细的错误信息
        import traceback
        logging.error(traceback.format_exc())
        return False

# 主函数
def main():
    # 添加命令行参数
    parser = argparse.ArgumentParser(description='生成问答对')
    parser.add_argument('--input', help='输入文件路径', default='segmented_content.json')
    parser.add_argument('--output_dir', help='输出目录', default='.')
    args = parser.parse_args()
    
    logging.info("开始生成Q/A指令对...")
    logging.info(f"输入文件: {args.input}, 输出目录: {args.output_dir}")
    
    # 1. 确保输出目录存在 - 在任何处理前先创建输出目录
    try:
        output_dir = Path(args.output_dir)
        # 使用os.makedirs确保目录创建
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"确保输出目录存在: {output_dir}")
    except Exception as e:
        logging.error(f"创建输出目录失败: {str(e)}")
        return 1
    
    # 2. 验证输入文件是否存在
    if not os.path.exists(args.input):
        logging.error(f"输入文件不存在: {args.input}")
        return 1
    
    # 3. 加载分段内容
    segments = load_segments(args.input)
    if not segments:
        logging.error("无法加载分段内容，退出程序")
        return 1
    
    logging.info(f"成功加载了 {len(segments)} 个文本段落")
    
    # 4. 为每个段落生成多样化问题
    all_qa_pairs = []
    for segment in segments:
        segment_qa_pairs = generate_diverse_questions(segment)
        all_qa_pairs.extend(segment_qa_pairs)
        logging.info(f"为段落 {segment['segment_id']} 生成了 {len(segment_qa_pairs)} 个问答对")
    
    # 5. 验证所有问答对
    validated_pairs, fixed_count = validate_qa_pairs(all_qa_pairs, segments)
    
    # 6. 创建ChatGLM格式的对话数据
    chatglm_conversations = []
    for pair in validated_pairs:
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["user"]},
            {"role": "assistant", "content": pair["assistant"]}
        ]
        meta = {
            "conversations": conversation,
            "segment_id": pair["segment_id"],
            "dimension": pair["dimension"],
            "difficulty": pair["difficulty"]
        }
        # 如果有主题信息，也添加进去
        if "topic" in pair:
            meta["topic"] = pair["topic"]
            
        chatglm_conversations.append(meta)
    
    # 7. 统计分布信息
    # 统计维度分布
    dimension_counts = {}
    for pair in validated_pairs:
        dimension = pair["dimension"]
        dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1
    
    # 统计难度分布
    difficulty_counts = {}
    for pair in validated_pairs:
        difficulty = pair["difficulty"]
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    # 8. 保存结果
    stats = {
        "修复了问题或答案": fixed_count,
        "思考维度分布": dimension_counts,
        "难度分布": difficulty_counts,
        "覆盖段落数": f"{len(set(pair['segment_id'] for pair in validated_pairs))}/{len(segments)}"
    }
    
    if save_to_files(validated_pairs, chatglm_conversations, args.output_dir, stats):
        return 0
    else:
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 