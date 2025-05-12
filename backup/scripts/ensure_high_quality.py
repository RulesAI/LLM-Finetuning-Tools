import json
import re
import logging
import tiktoken
import random
from pathlib import Path
from collections import Counter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quality_check.log"),
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

class QualityChecker:
    def __init__(self):
        # 质量评分标准
        self.quality_criteria = {
            "question_length": (10, 50),  # 问题长度范围(min, max)
            "answer_length": (100, 800),  # 答案长度范围(min, max)
            "question_token_count": (5, 25),  # 问题token数范围
            "answer_token_count": (50, 350),  # 答案token数范围
            "question_ends_with_mark": True,  # 问题是否以问号结尾
            "invalid_chars": ['》', '《', '\u0001', '==='],  # 无效字符列表
            "topic_blacklist": ['null', 'undefined', 'none']  # 主题黑名单
        }
        
        # 常见问题清单
        self.common_issues = {
            "incomplete_question": 0,  # 不完整的问题
            "invalid_chars": 0,        # 含有无效字符
            "too_short_answer": 0,     # 答案过短
            "too_long_answer": 0,      # 答案过长
            "no_question_mark": 0,     # 没有问号
            "invalid_topic": 0,        # 无效主题
            "invalid_dimension": 0,    # 无效维度
            "duplicate_qa": 0,         # 重复问答对
            "topic_mismatch": 0,       # 主题与内容不匹配
            "question_answer_mismatch": 0  # 问题与答案不匹配
        }
        
        # 初始化问题和答案缓存，用于检测重复
        self.question_cache = set()
        self.answer_cache = set()
        
        # 有效维度和难度
        self.valid_dimensions = {"释义类", "应用类", "对比类", "推导类", "纠错类"}
        self.valid_difficulties = {"basic", "medium", "advanced", "expert", "challenge"}
        
        # 供应链领域关键词(用于相关性检查)
        self.domain_keywords = [
            "供应链", "物流", "仓库", "运输", "库存", "WMS", "TMS", "ERP", "执行", "技术",
            "自动化", "机器人", "智能", "数字化", "平台", "系统", "管理", "优化", "预测",
            "可视化", "跟踪", "监控", "调度", "分拣", "配送", "履约", "计划", "协同"
        ]

    def check_question_quality(self, question):
        """检查问题质量并返回得分和问题"""
        if not question:
            self.common_issues["incomplete_question"] += 1
            return 0, ["问题为空"]
        
        issues = []
        score = 100  # 初始满分
        
        # 检查长度
        q_len = len(question)
        min_len, max_len = self.quality_criteria["question_length"]
        if q_len < min_len:
            score -= 30
            issues.append(f"问题过短({q_len}<{min_len})")
            self.common_issues["incomplete_question"] += 1
        elif q_len > max_len * 1.5:
            score -= 10
            issues.append(f"问题过长({q_len}>{max_len*1.5})")
        
        # 检查token数
        q_tokens = count_tokens(question)
        min_tokens, max_tokens = self.quality_criteria["question_token_count"]
        if q_tokens < min_tokens:
            score -= 20
            issues.append(f"问题token数过少({q_tokens}<{min_tokens})")
        elif q_tokens > max_tokens * 1.5:
            score -= 10
            issues.append(f"问题token数过多({q_tokens}>{max_tokens*1.5})")
        
        # 检查是否以问号结尾
        if not question.endswith('？') and not question.endswith('?'):
            score -= 15
            issues.append("问题不以问号结尾")
            self.common_issues["no_question_mark"] += 1
        
        # 检查无效字符
        for char in self.quality_criteria["invalid_chars"]:
            if char in question:
                score -= 25
                issues.append(f"问题包含无效字符 '{char}'")
                self.common_issues["invalid_chars"] += 1
                break
        
        # 检查重复性
        if question in self.question_cache:
            score -= 50
            issues.append("问题与现有问题重复")
            self.common_issues["duplicate_qa"] += 1
        else:
            self.question_cache.add(question)
        
        return score, issues

    def check_answer_quality(self, answer):
        """检查答案质量并返回得分和问题"""
        if not answer:
            self.common_issues["too_short_answer"] += 1
            return 0, ["答案为空"]
        
        issues = []
        score = 100  # 初始满分
        
        # 检查长度
        a_len = len(answer)
        min_len, max_len = self.quality_criteria["answer_length"]
        if a_len < min_len:
            score -= 30
            issues.append(f"答案过短({a_len}<{min_len})")
            self.common_issues["too_short_answer"] += 1
        elif a_len > max_len * 1.5:
            score -= 15
            issues.append(f"答案过长({a_len}>{max_len*1.5})")
            self.common_issues["too_long_answer"] += 1
        
        # 检查token数
        a_tokens = count_tokens(answer)
        min_tokens, max_tokens = self.quality_criteria["answer_token_count"]
        if a_tokens < min_tokens:
            score -= 20
            issues.append(f"答案token数过少({a_tokens}<{min_tokens})")
        elif a_tokens > max_tokens * 1.5:
            score -= 10
            issues.append(f"答案token数过多({a_tokens}>{max_tokens*1.5})")
        
        # 检查无效字符
        for char in self.quality_criteria["invalid_chars"]:
            if char in answer:
                score -= 25
                issues.append(f"答案包含无效字符 '{char}'")
                self.common_issues["invalid_chars"] += 1
                break
        
        # 检查重复性 (使用答案的前50个字符来降低误判)
        answer_signature = answer[:min(50, len(answer))]
        if answer_signature in self.answer_cache:
            score -= 20
            issues.append("答案与现有答案相似")
            self.common_issues["duplicate_qa"] += 1
        else:
            self.answer_cache.add(answer_signature)
        
        return score, issues
    
    def check_question_answer_relevance(self, question, answer, topic):
        """检查问题和答案的相关性"""
        if not question or not answer:
            return 0, ["问题或答案为空"]
        
        issues = []
        score = 100  # 初始满分
        
        # 提取问题中的关键词
        question_words = set(re.findall(r'\w+', question.lower()))
        # 去除停用词
        stop_words = {"的", "是", "在", "与", "和", "有", "什么", "如何", "为什么", "哪些", "多少", "为何"}
        question_keywords = question_words - stop_words
        
        # 提取主题关键词
        topic_words = set(re.findall(r'\w+', topic.lower())) if topic else set()
        
        # 检查主题词是否出现在答案中
        topic_match_count = 0
        for word in topic_words:
            if len(word) > 1 and word in answer.lower():
                topic_match_count += 1
        
        topic_match_ratio = topic_match_count / len(topic_words) if topic_words else 0
        if topic_match_ratio < 0.3:
            score -= 30
            issues.append(f"答案中未充分包含主题'{topic}'相关内容")
            self.common_issues["topic_mismatch"] += 1
        
        # 检查问题关键词是否在答案中有对应
        keyword_match_count = 0
        for word in question_keywords:
            if len(word) > 1 and word in answer.lower():
                keyword_match_count += 1
        
        # 计算匹配率
        match_ratio = keyword_match_count / len(question_keywords) if question_keywords else 0
        if match_ratio < 0.3:
            score -= 40
            issues.append("问题与答案内容不匹配")
            self.common_issues["question_answer_mismatch"] += 1
        elif match_ratio < 0.5:
            score -= 20
            issues.append("问题与答案相关性较低")
        
        # 检查特殊情况：问题询问定义/解释但答案未提供
        if re.search(r'(什么是|定义|解释|概念)', question) and "是" not in answer[:100]:
            score -= 15
            issues.append("问题询问定义但答案未提供直接解释")
        
        # 检查特殊情况：问到"为什么"但答案未包含原因解释
        if "为什么" in question and not any(w in answer[:200] for w in ["因为", "原因", "由于"]):
            score -= 15
            issues.append("问题询问原因但答案未提供清晰解释")
        
        # 检查是否出现"责任归属"与"自动驾驶"不匹配的情况
        if "责任归属" in topic and "自动驾驶" not in topic and "自动驾驶" in answer:
            score -= 25
            issues.append("主题'责任归属'可能应为'自动驾驶技术'的一部分")
            self.common_issues["topic_mismatch"] += 1
        
        return score, issues
        
    def check_metadata_quality(self, metadata):
        """检查元数据质量"""
        score = 100
        issues = []
        
        # 检查主题
        if "topic" in metadata:
            topic = metadata["topic"]
            if not topic or topic.lower() in self.quality_criteria["topic_blacklist"]:
                score -= 20
                issues.append(f"无效主题: {topic}")
                self.common_issues["invalid_topic"] += 1
            
            # 检查主题格式问题
            if topic.startswith("的") or topic.endswith("的因素") or topic == "责任归属":
                score -= 15
                issues.append(f"主题格式可能不正确: {topic}")
                self.common_issues["invalid_topic"] += 1
        
        # 检查维度
        if "dimension" in metadata:
            dimension = metadata["dimension"]
            if dimension not in self.valid_dimensions:
                score -= 10
                issues.append(f"无效维度: {dimension}")
                self.common_issues["invalid_dimension"] += 1
        
        # 检查难度
        if "difficulty" in metadata:
            difficulty = metadata["difficulty"]
            if difficulty not in self.valid_difficulties:
                score -= 10
                issues.append(f"无效难度: {difficulty}")
        
        return score, issues

    def evaluate_pair(self, pair, format_type="json"):
        """评估单个问答对的质量"""
        question = None
        answer = None
        metadata = {}
        
        # 根据不同格式提取问题、答案和元数据
        if format_type == "jsonl":
            # ChatGLM格式
            conversations = pair.get("conversations", [])
            for conv in conversations:
                if conv.get("role") == "user":
                    question = conv.get("content", "")
                elif conv.get("role") == "assistant":
                    answer = conv.get("content", "")
            
            # 提取元数据
            for key in ["topic", "dimension", "difficulty", "segment_id"]:
                if key in pair:
                    metadata[key] = pair[key]
        else:
            # 标准JSON格式
            question = pair.get("user", "")
            answer = pair.get("assistant", "")
            
            # 提取元数据
            for key in ["topic", "dimension", "difficulty", "segment_id"]:
                if key in pair:
                    metadata[key] = pair[key]
        
        # 检查问题和答案质量
        q_score, q_issues = self.check_question_quality(question)
        a_score, a_issues = self.check_answer_quality(answer)
        m_score, m_issues = self.check_metadata_quality(metadata)
        
        # 检查问题和答案的相关性
        r_score, r_issues = self.check_question_answer_relevance(
            question, answer, metadata.get("topic", "")
        )
        
        # 计算总分 (加权平均)
        # 增加了问答相关性的权重，从而更严格地筛选问答对
        total_score = q_score * 0.3 + a_score * 0.3 + m_score * 0.1 + r_score * 0.3
        all_issues = q_issues + a_issues + m_issues + r_issues
        
        result = {
            "question": question,
            "answer": answer,
            "metadata": metadata,
            "quality": {
                "total_score": round(total_score, 1),
                "question_score": q_score,
                "answer_score": a_score,
                "metadata_score": m_score,
                "relevance_score": r_score,
                "issues": all_issues
            }
        }
        
        return result

    def analyze_file(self, file_path):
        """分析文件中的问答对质量"""
        logging.info(f"开始分析文件: {file_path}")
        
        try:
            # 确定文件格式
            format_type = "jsonl" if file_path.endswith(".jsonl") else "json"
            
            # 加载文件
            with open(file_path, 'r', encoding='utf-8') as f:
                if format_type == "jsonl":
                    # 按行读取JSONL
                    pairs = []
                    for line in f:
                        if line.strip():
                            try:
                                pairs.append(json.loads(line))
                            except json.JSONDecodeError:
                                logging.warning(f"解析JSONL行失败: {line[:50]}...")
                else:
                    # 整体读取JSON
                    try:
                        pairs = json.load(f)
                    except json.JSONDecodeError:
                        logging.error(f"解析JSON文件失败: {file_path}")
                        return None
            
            # 评估每个问答对
            results = []
            for pair in pairs:
                result = self.evaluate_pair(pair, format_type)
                results.append(result)
            
            # 统计结果
            scores = [r["quality"]["total_score"] for r in results]
            average_score = sum(scores) / len(scores) if scores else 0
            
            # 按得分分布
            score_distribution = {
                "excellent": len([s for s in scores if s >= 90]),
                "good": len([s for s in scores if 75 <= s < 90]),
                "fair": len([s for s in scores if 60 <= s < 75]),
                "poor": len([s for s in scores if s < 60])
            }
            
            report = {
                "filename": file_path,
                "total_pairs": len(pairs),
                "average_score": round(average_score, 1),
                "score_distribution": score_distribution,
                "common_issues": self.common_issues
            }
            
            logging.info(f"文件 {file_path} 分析完成, 平均得分: {round(average_score, 1)}")
            
            return {
                "results": results,
                "report": report
            }
        
        except Exception as e:
            logging.error(f"分析文件时出错: {str(e)}")
            return None

    def extract_high_quality_pairs(self, results, min_score=85, max_count=None, balanced=True):
        """从评估结果中提取高质量问答对"""
        if not results:
            return []
        
        # 按得分排序
        sorted_results = sorted(results, key=lambda r: r["quality"]["total_score"], reverse=True)
        
        # 过滤出高于最低分数的结果
        high_quality = [r for r in sorted_results if r["quality"]["total_score"] >= min_score]
        
        # 如果开启平衡模式，尝试平衡不同难度和维度的分布
        if balanced and high_quality:
            # 按难度和维度统计
            by_difficulty = {}
            by_dimension = {}
            
            for r in high_quality:
                difficulty = r["metadata"].get("difficulty", "unknown")
                dimension = r["metadata"].get("dimension", "unknown")
                
                by_difficulty[difficulty] = by_difficulty.get(difficulty, 0) + 1
                by_dimension[dimension] = by_dimension.get(dimension, 0) + 1
            
            # 检查是否有某些类别严重不足
            min_difficulty_count = min(by_difficulty.values()) if by_difficulty else 0
            min_dimension_count = min(by_dimension.values()) if by_dimension else 0
            
            # 如果某类别数量不足，尝试从较低分数的结果中添加
            if min_difficulty_count < max(by_difficulty.values()) / 3 or min_dimension_count < max(by_dimension.values()) / 3:
                # 找出数量不足的类别
                underrepresented_difficulties = [d for d, c in by_difficulty.items() if c <= min_difficulty_count]
                underrepresented_dimensions = [d for d, c in by_dimension.items() if c <= min_dimension_count]
                
                # 从所有结果中筛选这些类别的高分结果
                additional_results = []
                
                for r in sorted_results:
                    if r in high_quality:
                        continue
                    
                    difficulty = r["metadata"].get("difficulty", "unknown")
                    dimension = r["metadata"].get("dimension", "unknown")
                    score = r["quality"]["total_score"]
                    
                    # 如果是数量不足的类别，且分数不太低，则考虑添加
                    if (difficulty in underrepresented_difficulties or dimension in underrepresented_dimensions) and score >= min_score - 10:
                        additional_results.append(r)
                
                # 添加额外结果，但仍然按照分数排序
                additional_results.sort(key=lambda r: r["quality"]["total_score"], reverse=True)
                high_quality.extend(additional_results[:20])  # 最多添加20个额外结果
                
                # 重新排序
                high_quality.sort(key=lambda r: r["quality"]["total_score"], reverse=True)
        
        # 应用最大数量限制
        if max_count and len(high_quality) > max_count:
            return high_quality[:max_count]
        
        return high_quality

    def save_high_quality_pairs(self, pairs, output_file, format_type="jsonl"):
        """将高质量问答对保存到文件"""
        try:
            if format_type == "jsonl":
                # 转换为ChatGLM格式
                chatglm_pairs = []
                for pair in pairs:
                    conversation = [
                        {"role": "system", "content": "你是一个专业的供应链技术领域顾问，擅长解释各种供应链执行技术的概念、应用场景和发展趋势。请基于你的专业知识回答问题。"},
                        {"role": "user", "content": pair["question"]},
                        {"role": "assistant", "content": pair["answer"]}
                    ]
                    
                    meta = {
                        "conversations": conversation,
                        "quality_score": pair["quality"]["total_score"]
                    }
                    
                    # 添加元数据
                    for key, value in pair["metadata"].items():
                        meta[key] = value
                    
                    chatglm_pairs.append(meta)
                
                # 保存为JSONL格式
                with open(output_file, 'w', encoding='utf-8') as f:
                    for pair in chatglm_pairs:
                        f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            else:
                # 转换为标准JSON格式
                json_pairs = []
                for pair in pairs:
                    json_pair = {
                        "system": "你是一个专业的供应链技术领域顾问，擅长解释各种供应链执行技术的概念、应用场景和发展趋势。请基于你的专业知识回答问题。",
                        "user": pair["question"],
                        "assistant": pair["answer"],
                        "quality_score": pair["quality"]["total_score"]
                    }
                    
                    # 添加元数据
                    for key, value in pair["metadata"].items():
                        json_pair[key] = value
                    
                    json_pairs.append(json_pair)
                
                # 保存为JSON格式
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_pairs, f, ensure_ascii=False, indent=2)
            
            logging.info(f"已将 {len(pairs)} 个高质量问答对保存到 {output_file}")
            return True
        
        except Exception as e:
            logging.error(f"保存高质量问答对时出错: {str(e)}")
            return False

    def save_quality_report(self, report, output_file="quality_report.json"):
        """保存质量分析报告"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logging.info(f"质量分析报告已保存到 {output_file}")
            return True
        
        except Exception as e:
            logging.error(f"保存质量分析报告时出错: {str(e)}")
            return False

def main():
    logging.info("开始检查问答对质量...")
    
    # 创建质量检查器
    checker = QualityChecker()
    
    # 需要分析的文件列表
    files_to_check = [
        "qa_instructions_chatglm_robust.jsonl",
        "qa_instructions_chatglm_fixed_improved.jsonl",
        "qa_instructions_robust.json",
        "qa_instructions_fixed_improved.json"
    ]
    
    all_high_quality_pairs = []
    
    # 分析每个文件
    for file_path in files_to_check:
        if not Path(file_path).exists():
            logging.warning(f"文件 {file_path} 不存在，跳过")
            continue
        
        # 分析文件
        result = checker.analyze_file(file_path)
        
        if result:
            # 保存分析报告
            report_file = f"quality_report_{Path(file_path).stem}.json"
            checker.save_quality_report(result["report"], report_file)
            
            # 提取高质量问答对
            high_quality = checker.extract_high_quality_pairs(
                result["results"], 
                min_score=85,  # 最低质量分数
                max_count=None,  # 不限制每个文件的数量
                balanced=True    # 平衡各维度和难度
            )
            
            all_high_quality_pairs.extend(high_quality)
    
    # 进一步选择最终的高质量问答对集合
    if all_high_quality_pairs:
        # 随机打乱，确保多样性
        random.shuffle(all_high_quality_pairs)
        
        # 按分数重新排序
        all_high_quality_pairs.sort(key=lambda x: x["quality"]["total_score"], reverse=True)
        
        # 限制最终数量为200个
        final_count = min(200, len(all_high_quality_pairs))
        final_selection = all_high_quality_pairs[:final_count]
        
        # 保存最终高质量问答对
        checker.save_high_quality_pairs(
            final_selection, 
            "high_quality_qa.jsonl", 
            format_type="jsonl"
        )
        
        checker.save_high_quality_pairs(
            final_selection, 
            "high_quality_qa.json", 
            format_type="json"
        )
        
        logging.info(f"已创建包含 {final_count} 个高质量问答对的最终训练集")
    else:
        logging.warning("未找到任何高质量问答对")

if __name__ == "__main__":
    main() 