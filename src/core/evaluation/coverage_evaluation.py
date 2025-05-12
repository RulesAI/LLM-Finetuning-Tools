import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

#!/usr/bin/env python3
import json
import os
import argparse
import logging
import yaml
import re
import math
import numpy as np
from collections import defaultdict

# 数据处理和哈希
from datasketch import MinHash, MinHashLSH
import jieba
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# 知识点提取
import spacy

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# API调用
from openai import OpenAI
from src.core.utils.model_utils import ModelCaller

# 设置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("coverage_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("coverage_evaluation")

# 加载配置
def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        raise

# 文档处理函数
def load_document(file_path):
    """加载文档内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"成功加载文档: {file_path}")
        return content
    except Exception as e:
        logger.error(f"加载文档失败: {str(e)}")
        raise

def load_qa_pairs(file_path):
    """加载问答对数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 计算问答对总数
        total_qa_pairs = sum(len(topic.get('qa_pairs', [])) for topic in data)
        logger.info(f"成功加载问答对: {file_path}, 共{len(data)}个主题, {total_qa_pairs}个问答对")
        return data
    except Exception as e:
        logger.error(f"加载问答对数据失败: {str(e)}")
        raise

def split_document(content, min_length=50):
    """
    将文档分割为段落
    """
    # 首先尝试按双换行符分割
    paragraphs = re.split(r'\n\s*\n', content)
    
    # 过滤掉太短的段落
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) >= min_length]
    
    # 如果段落太少，尝试按单换行符分割
    if len(paragraphs) < 5:
        paragraphs = re.split(r'\n', content)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) >= min_length]
    
    logger.info(f"文档分割完成，共{len(paragraphs)}个段落")
    return paragraphs

# ========================== 哈希评估模块 ==========================

def compute_minhash(segments, num_perm=128):
    """为文档段落计算MinHash"""
    doc_hashes = {}
    
    for i, segment in enumerate(segments):
        # 分词
        tokens = list(jieba.cut(re.sub(r'\s+', ' ', segment)))
        
        # 创建MinHash
        mh = MinHash(num_perm=num_perm)
        for token in tokens:
            mh.update(token.encode('utf-8'))
        
        doc_hashes[i] = {
            'minhash': mh,
            'content': segment,
            'length': len(segment)
        }
    
    logger.info(f"计算了{len(doc_hashes)}个段落的MinHash值")
    return doc_hashes

def compute_qa_minhash(qa_data, num_perm=128):
    """为问答对计算MinHash"""
    qa_hashes = []
    
    for topic_idx, topic in enumerate(qa_data):
        topic_name = topic.get('topic', f'未命名主题_{topic_idx}')
        
        for qa_idx, qa in enumerate(topic.get('qa_pairs', [])):
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            
            # 合并问题和答案进行哈希
            qa_text = question + ' ' + answer
            tokens = list(jieba.cut(re.sub(r'\s+', ' ', qa_text)))
            
            # 创建MinHash
            mh = MinHash(num_perm=num_perm)
            for token in tokens:
                mh.update(token.encode('utf-8'))
            
            qa_hashes.append({
                'minhash': mh,
                'topic': topic_name,
                'question': question,
                'answer': answer,
                'qa_idx': qa_idx,
                'topic_idx': topic_idx
            })
    
    logger.info(f"计算了{len(qa_hashes)}个问答对的MinHash值")
    return qa_hashes

def compute_lsh_coverage(doc_hashes, qa_hashes, threshold=0.3):
    """计算文档覆盖统计"""
    # 构建LSH索引
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    
    # 将文档段落添加到LSH
    for i, doc_data in doc_hashes.items():
        lsh.insert(f"doc_{i}", doc_data['minhash'])
    
    # 查询每个问答对，找到匹配的段落
    coverage_stats = {
        'covered_paragraphs': set(),
        'paragraph_coverage': defaultdict(list),
        'qa_matches': defaultdict(list)
    }
    
    for qa_idx, qa_data in enumerate(qa_hashes):
        result = lsh.query(qa_data['minhash'])
        
        # 记录匹配结果
        if result:
            for doc_id in result:
                # 从doc_1中提取数字1
                para_idx = int(doc_id.split('_')[1])
                coverage_stats['covered_paragraphs'].add(para_idx)
                coverage_stats['paragraph_coverage'][para_idx].append(qa_idx)
                coverage_stats['qa_matches'][qa_idx].append(para_idx)
    
    # 计算覆盖率
    coverage_stats['total_paragraphs'] = len(doc_hashes)
    coverage_stats['coverage_rate'] = len(coverage_stats['covered_paragraphs']) / len(doc_hashes)
    
    # 计算问答对的匹配情况
    coverage_stats['qa_with_match'] = len([qa for qa in coverage_stats['qa_matches'] if coverage_stats['qa_matches'][qa]])
    coverage_stats['qa_match_rate'] = coverage_stats['qa_with_match'] / len(qa_hashes) if qa_hashes else 0
    
    logger.info(f"文档段落总数: {coverage_stats['total_paragraphs']}, "
               f"被覆盖段落数: {len(coverage_stats['covered_paragraphs'])}, "
               f"覆盖率: {coverage_stats['coverage_rate']:.2f}")
    
    return coverage_stats

def generate_coverage_heatmap(segments, coverage_stats, output_path):
    """生成覆盖度热图"""
    # 准备热图数据
    paragraph_indices = list(range(len(segments)))
    coverage_values = []
    
    for i in paragraph_indices:
        coverage_values.append(len(coverage_stats['paragraph_coverage'].get(i, [])))
    
    # 创建颜色映射
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'lightblue', 'blue', 'darkblue'], N=256)
    
    # 准备热图数据矩阵
    # 确定热图的行列数
    num_rows = math.ceil(len(segments) / 4)  # 每行最多4个段落
    heatmap_data = np.zeros((num_rows, 4))
    
    for i, value in enumerate(coverage_values):
        row = i // 4
        col = i % 4
        if row < num_rows and col < 4:
            heatmap_data[row, col] = value
    
    # 创建热图
    plt.figure(figsize=(12, max(6, num_rows * 0.5)))
    ax = sns.heatmap(heatmap_data, cmap=cmap, annot=True, fmt=".0f", linewidths=.5)
    
    plt.title("文档段落覆盖度热图")
    plt.xlabel("段落列索引")
    plt.ylabel("段落行索引")
    
    # 保存热图
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"覆盖度热图已保存到: {output_path}")
    return output_path

def hash_based_evaluation(doc_content, qa_data, output_dir, threshold=0.3):
    """执行基于哈希的评估"""
    # 文档分段
    segments = split_document(doc_content)
    
    # 计算段落哈希值
    doc_hashes = compute_minhash(segments)
    
    # 准备问答对数据
    qa_pairs_flat = []
    for topic in qa_data:
        topic_name = topic.get('topic', '未知主题')
        for qa in topic.get('qa_pairs', []):
            qa['topic'] = topic_name
            qa_pairs_flat.append(qa)
    
    # 计算问答对哈希值
    qa_hashes = compute_qa_minhash(qa_data)
    
    # 构建LSH索引和查询
    coverage_stats = compute_lsh_coverage(doc_hashes, qa_hashes, threshold)
    
    # 生成热图
    heatmap_path = os.path.join(output_dir, 'coverage_heatmap.png')
    generate_coverage_heatmap(segments, coverage_stats, heatmap_path)
    
    # 准备结果
    hash_results = {
        'coverage_rate': coverage_stats['coverage_rate'],
        'covered_paragraphs': len(coverage_stats['covered_paragraphs']),
        'total_paragraphs': coverage_stats['total_paragraphs'],
        'qa_with_match': coverage_stats['qa_with_match'],
        'qa_match_rate': coverage_stats['qa_match_rate'],
        'paragraph_details': [
            {
                'index': i,
                'content': segments[i][:100] + '...',  # 只保存前100个字符
                'matched_qa_count': len(coverage_stats['paragraph_coverage'].get(i, [])),
                'matched_qa_indices': coverage_stats['paragraph_coverage'].get(i, [])
            }
            for i in range(len(segments))
        ],
        'qa_details': [
            {
                'index': i,
                'question': qa_hashes[i]['question'],
                'topic': qa_hashes[i]['topic'],
                'matched_paragraphs': coverage_stats['qa_matches'].get(i, [])
            }
            for i in range(len(qa_hashes))
        ]
    }
    
    # 保存结果
    hash_results_path = os.path.join(output_dir, 'hash_evaluation_results.json')
    with open(hash_results_path, 'w', encoding='utf-8') as f:
        json.dump(hash_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"哈希评估结果已保存到: {hash_results_path}")
    
    return hash_results, segments, coverage_stats

# ========================== 知识点评估模块 ==========================

def extract_knowledge_points(content, nlp, model_caller=None):
    """提取文档中的知识点"""
    # 使用spaCy进行初步处理
    doc = nlp(content)
    
    # 基础知识点提取 - 使用实体和关键短语
    basic_points = []
    
    # 提取命名实体
    for ent in doc.ents:
        basic_points.append({
            'text': ent.text,
            'type': ent.label_,
            'source': 'entity'
        })
    
    # 提取主题短语和名词短语
    for chunk in doc.noun_chunks:
        if len(chunk.text.strip()) > 3:  # 过滤太短的短语
            basic_points.append({
                'text': chunk.text,
                'type': 'NOUN_PHRASE',
                'source': 'noun_chunk'
            })
    
    # 如果提供了模型调用器，使用LLM提取更多知识点
    advanced_points = []
    if model_caller:
        try:
            # 分段进行LLM处理，避免内容过长
            paragraphs = split_document(content)
            
            for i, para in enumerate(paragraphs):
                if len(para) < 50:  # 跳过太短的段落
                    continue
                
                prompt = f"""
请从以下供应链技术文档段落中提取关键知识点。
每个知识点应该是文档中明确表达的概念、技术、方法、趋势或见解。
格式要求：
1. 每个知识点用一句简洁的话表达
2. 不要添加编号
3. 不要添加解释
4. 直接输出知识点列表，每行一个
5. 如果段落中没有明确的知识点，请输出"无明确知识点"

文档段落:
{para}
"""
                
                result = model_caller.call_model('knowledge_extraction', prompt)
                
                # 处理结果
                points = [line.strip() for line in result.split('\n') if line.strip() and '无明确知识点' not in line]
                
                for point in points:
                    advanced_points.append({
                        'text': point,
                        'type': 'LLM_EXTRACTED',
                        'source': f'paragraph_{i}',
                        'paragraph_idx': i
                    })
            
            logger.info(f"LLM提取了{len(advanced_points)}个知识点")
        
        except Exception as e:
            logger.error(f"LLM知识点提取失败: {str(e)}")
    
    # 合并基础知识点和高级知识点
    all_points = basic_points + advanced_points
    
    # 去重
    unique_points = []
    seen_texts = set()
    for point in all_points:
        normalized_text = re.sub(r'\s+', ' ', point['text'].lower())
        if normalized_text not in seen_texts:
            seen_texts.add(normalized_text)
            unique_points.append(point)
    
    logger.info(f"共提取出{len(unique_points)}个唯一知识点")
    return unique_points

def prioritize_knowledge_points(knowledge_points, hash_results):
    """根据哈希评估结果，优先关注低覆盖区域的知识点"""
    # 获取低覆盖区域
    low_coverage_paragraphs = [
        p['index'] for p in hash_results['paragraph_details'] 
        if p['matched_qa_count'] == 0
    ]
    
    # 为知识点分配优先级
    for point in knowledge_points:
        if 'paragraph_idx' in point and point['paragraph_idx'] in low_coverage_paragraphs:
            point['priority'] = 'high'
        else:
            point['priority'] = 'normal'
    
    # 将高优先级的知识点排在前面
    prioritized_points = sorted(knowledge_points, key=lambda x: 0 if x.get('priority') == 'high' else 1)
    
    logger.info(f"优先级调整后，高优先级知识点数量: {sum(1 for p in prioritized_points if p.get('priority') == 'high')}")
    return prioritized_points

def map_qa_to_knowledge(qa_data, knowledge_points, nlp):
    """将问答对映射到知识点"""
    # 准备映射结果
    mapping_results = {
        'mappings': [],
        'knowledge_coverage': defaultdict(list),
        'qa_knowledge_points': defaultdict(list)
    }
    
    # 文本相似度函数
    def text_similarity(text1, text2, nlp):
        doc1 = nlp(text1.lower())
        doc2 = nlp(text2.lower())
        return doc1.similarity(doc2)
    
    # 问答对计数
    qa_count = 0
    
    # 遍历所有问答对
    for topic_idx, topic in enumerate(qa_data):
        topic_name = topic.get('topic', f'未命名主题_{topic_idx}')
        
        for qa_idx, qa in enumerate(topic.get('qa_pairs', [])):
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            qa_text = question + ' ' + answer
            
            # 为每个问答对分配唯一ID
            qa_id = f"{topic_idx}_{qa_idx}"
            
            # 匹配知识点
            matched_points = []
            
            for k_idx, point in enumerate(knowledge_points):
                # 计算相似度
                similarity = text_similarity(qa_text, point['text'], nlp)
                
                # 如果相似度高于阈值，认为是匹配的
                if similarity > 0.5:
                    matched_points.append({
                        'knowledge_idx': k_idx,
                        'knowledge_text': point['text'],
                        'similarity': similarity,
                        'priority': point.get('priority', 'normal')
                    })
                    
                    # 更新知识点覆盖信息
                    mapping_results['knowledge_coverage'][k_idx].append({
                        'qa_id': qa_id,
                        'similarity': similarity
                    })
            
            # 更新问答对的知识点信息
            mapping_results['qa_knowledge_points'][qa_id] = [mp['knowledge_idx'] for mp in matched_points]
            
            # 添加到映射结果
            mapping_results['mappings'].append({
                'qa_id': qa_id,
                'topic': topic_name,
                'question': question,
                'answer': answer[:100] + '...',  # 只保存前100个字符
                'matched_points': matched_points
            })
            
            qa_count += 1
    
    logger.info(f"完成了{qa_count}个问答对到知识点的映射")
    return mapping_results

def calculate_weighted_coverage(mapping_results, knowledge_points):
    """计算加权覆盖率"""
    # 计算每个知识点的权重
    knowledge_weights = {}
    for i, point in enumerate(knowledge_points):
        # 高优先级知识点权重更高
        if point.get('priority') == 'high':
            knowledge_weights[i] = 1.5
        else:
            knowledge_weights[i] = 1.0
    
    # 计算覆盖的知识点数量和权重总和
    covered_points = set()
    covered_weight = 0
    total_weight = sum(knowledge_weights.values())
    
    for k_idx in mapping_results['knowledge_coverage']:
        covered_points.add(k_idx)
        covered_weight += knowledge_weights.get(k_idx, 1.0)
    
    # 计算覆盖率
    simple_coverage = len(covered_points) / len(knowledge_points) if knowledge_points else 0
    weighted_coverage = covered_weight / total_weight if total_weight > 0 else 0
    
    # 计算每个主题的覆盖情况
    topic_coverage = defaultdict(set)
    for mapping in mapping_results['mappings']:
        topic = mapping['topic']
        for point in mapping['matched_points']:
            topic_coverage[topic].add(point['knowledge_idx'])
    
    topic_coverage_stats = {
        topic: {
            'covered_points': len(points),
            'coverage_rate': len(points) / len(knowledge_points) if knowledge_points else 0
        }
        for topic, points in topic_coverage.items()
    }
    
    # 返回结果
    coverage_stats = {
        'simple_coverage': simple_coverage,
        'weighted_coverage': weighted_coverage,
        'covered_points': len(covered_points),
        'total_points': len(knowledge_points),
        'topic_coverage': topic_coverage_stats
    }
    
    logger.info(f"知识点总数: {len(knowledge_points)}, "
               f"覆盖知识点数: {len(covered_points)}, "
               f"简单覆盖率: {simple_coverage:.2f}, "
               f"加权覆盖率: {weighted_coverage:.2f}")
    
    return coverage_stats

def identify_knowledge_gaps(mapping_results, knowledge_points):
    """识别未覆盖的重要知识点"""
    # 获取所有已覆盖的知识点
    covered_points = set(mapping_results['knowledge_coverage'].keys())
    
    # 识别未覆盖的知识点
    uncovered_points = []
    for i, point in enumerate(knowledge_points):
        if i not in covered_points:
            uncovered_points.append({
                'index': i,
                'text': point['text'],
                'type': point.get('type', 'UNKNOWN'),
                'priority': point.get('priority', 'normal')
            })
    
    # 按优先级排序
    uncovered_points = sorted(uncovered_points, key=lambda x: 0 if x.get('priority') == 'high' else 1)
    
    # 计算高优先级未覆盖点
    high_priority_gaps = [p for p in uncovered_points if p.get('priority') == 'high']
    
    logger.info(f"发现{len(uncovered_points)}个未覆盖知识点，其中{len(high_priority_gaps)}个为高优先级")
    return {
        'all_gaps': uncovered_points,
        'high_priority_gaps': high_priority_gaps,
        'gap_count': len(uncovered_points),
        'high_priority_gap_count': len(high_priority_gaps)
    }

def knowledge_mapping_evaluation(doc_content, qa_data, knowledge_points, hash_results, model_caller, output_dir):
    """执行基于知识点映射的评估"""
    # 加载NLP模型
    try:
        nlp = spacy.load("zh_core_web_sm")
        logger.info("成功加载中文NLP模型")
    except:
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("成功加载英文NLP模型")
        except Exception as e:
            logger.error(f"加载NLP模型失败: {str(e)}")
            raise
    
    # 优先关注低覆盖区域
    prioritized_points = prioritize_knowledge_points(knowledge_points, hash_results)
    
    # 映射问答对到知识点
    mapping_results = map_qa_to_knowledge(qa_data, prioritized_points, nlp)
    
    # 计算加权覆盖率
    coverage_stats = calculate_weighted_coverage(mapping_results, prioritized_points)
    
    # 识别未覆盖的知识点
    knowledge_gaps = identify_knowledge_gaps(mapping_results, prioritized_points)
    
    # 保存结果
    knowledge_results = {
        'coverage_stats': coverage_stats,
        'knowledge_gaps': knowledge_gaps
    }
    
    knowledge_results_path = os.path.join(output_dir, 'knowledge_evaluation_results.json')
    with open(knowledge_results_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_results, f, ensure_ascii=False, indent=2)
    
    # 保存知识点列表
    knowledge_points_path = os.path.join(output_dir, 'knowledge_points.json')
    with open(knowledge_points_path, 'w', encoding='utf-8') as f:
        json.dump(prioritized_points, f, ensure_ascii=False, indent=2)
    
    logger.info(f"知识点评估结果已保存到: {knowledge_results_path}")
    logger.info(f"知识点列表已保存到: {knowledge_points_path}")
    
    return knowledge_results

def generate_report(output_dir, hash_results, knowledge_results, original_path, qa_path):
    """生成综合评估报告"""
    report = {
        'summary': {
            'original_document': original_path,
            'qa_pairs_file': qa_path,
            'hash_coverage_rate': hash_results['coverage_rate'],
            'knowledge_coverage_rate': knowledge_results['coverage_stats']['simple_coverage'],
            'weighted_knowledge_coverage': knowledge_results['coverage_stats']['weighted_coverage'],
            'total_paragraphs': hash_results['total_paragraphs'],
            'covered_paragraphs': hash_results['covered_paragraphs'],
            'total_knowledge_points': knowledge_results['coverage_stats']['total_points'],
            'covered_knowledge_points': knowledge_results['coverage_stats']['covered_points'],
            'uncovered_knowledge_points': knowledge_results['knowledge_gaps']['gap_count'],
            'high_priority_gaps': knowledge_results['knowledge_gaps']['high_priority_gap_count']
        },
        'recommendations': {
            'suggested_improvements': [],
            'uncovered_areas': []
        }
    }
    
    # 添加建议和未覆盖区域
    if knowledge_results['knowledge_gaps']['high_priority_gap_count'] > 0:
        report['recommendations']['suggested_improvements'].append(
            "建议增加以下高优先级知识点的问答对"
        )
        
        # 添加未覆盖的高优先级知识点
        for gap in knowledge_results['knowledge_gaps']['high_priority_gaps'][:5]:  # 最多显示5个
            report['recommendations']['uncovered_areas'].append({
                'knowledge_point': gap['text'],
                'priority': gap['priority']
            })
    
    # 根据覆盖率提供建议
    if hash_results['coverage_rate'] < 0.7:
        report['recommendations']['suggested_improvements'].append(
            f"文档段落覆盖率较低({hash_results['coverage_rate']:.2f})，建议增加问答对数量"
        )
    
    if knowledge_results['coverage_stats']['simple_coverage'] < 0.7:
        report['recommendations']['suggested_improvements'].append(
            f"知识点覆盖率较低({knowledge_results['coverage_stats']['simple_coverage']:.2f})，建议增加问答多样性"
        )
    
    # 保存报告
    report_path = os.path.join(output_dir, 'coverage_evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 生成文本报告
    text_report = f"""
# 文档覆盖度评估报告

## 总体情况
- 原始文档: {original_path}
- 问答对文件: {qa_path}
- 文档段落覆盖率: {hash_results['coverage_rate']:.2f} ({hash_results['covered_paragraphs']}/{hash_results['total_paragraphs']})
- 知识点覆盖率: {knowledge_results['coverage_stats']['simple_coverage']:.2f} ({knowledge_results['coverage_stats']['covered_points']}/{knowledge_results['coverage_stats']['total_points']})
- 加权知识点覆盖率: {knowledge_results['coverage_stats']['weighted_coverage']:.2f}

## 未覆盖的高优先级知识点
"""
    
    # 添加未覆盖知识点
    if knowledge_results['knowledge_gaps']['high_priority_gaps']:
        for i, gap in enumerate(knowledge_results['knowledge_gaps']['high_priority_gaps'][:10]):  # 最多显示10个
            text_report += f"- {i+1}. {gap['text']}\n"
    else:
        text_report += "- 无高优先级知识点缺口\n"
    
    text_report += "\n## 建议改进\n"
    for suggestion in report['recommendations']['suggested_improvements']:
        text_report += f"- {suggestion}\n"
    
    # 保存文本报告
    text_report_path = os.path.join(output_dir, 'coverage_evaluation_report.txt')
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write(text_report)
    
    logger.info(f"评估报告已保存到: {report_path} 和 {text_report_path}")
    
    return report_path, text_report_path

def main():
    parser = argparse.ArgumentParser(description='评估问答对对原始文档的覆盖度')
    parser.add_argument('--doc', required=True, help='原始文档路径')
    parser.add_argument('--qa', required=True, help='问答对JSON文件路径')
    parser.add_argument('--output', required=True, help='评估报告输出目录')
    parser.add_argument('--config', default='models_config.yaml', help='模型配置文件')
    parser.add_argument('--threshold', type=float, default=0.3, help='哈希相似度阈值')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化报告')
    args = parser.parse_args()
    
    # 确保输出目录存在
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # 加载配置
    config = load_config(args.config)
    
    # 初始化模型调用器
    model_caller = ModelCaller(config_path=args.config)
    
    # 加载数据
    original_doc = load_document(args.doc)
    qa_data = load_qa_pairs(args.qa)
    
    # 阶段1: 哈希评估
    logger.info("开始哈希评估阶段...")
    hash_results, segments, coverage_stats = hash_based_evaluation(
        original_doc, qa_data, args.output, args.threshold)
    
    # 提取知识点
    logger.info("提取文档知识点...")
    # 加载NLP模型
    try:
        nlp = spacy.load("zh_core_web_sm")
        logger.info("成功加载中文NLP模型")
    except:
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("成功加载英文NLP模型")
        except Exception as e:
            logger.error(f"加载NLP模型失败: {str(e)}")
            raise
    
    knowledge_points = extract_knowledge_points(original_doc, nlp, model_caller)
    
    # 阶段2: 知识点评估
    logger.info("开始知识点评估阶段...")
    knowledge_results = knowledge_mapping_evaluation(
        original_doc, qa_data, knowledge_points, hash_results, model_caller, args.output)
    
    # 生成综合报告
    logger.info("生成综合评估报告...")
    report_path, text_report_path = generate_report(
        args.output, hash_results, knowledge_results, args.doc, args.qa)
    
    logger.info(f"评估完成，报告已保存到: {args.output}")
    print(f"评估完成，报告已保存到: {args.output}")

if __name__ == "__main__":
    main() 