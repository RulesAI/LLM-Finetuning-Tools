import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

#!/usr/bin/env python3
import json
import os
import random
import argparse
from collections import defaultdict
from difflib import SequenceMatcher

def load_qa_data(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        if input_path.endswith('.jsonl'):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    return data

def flatten_qa_pairs(qa_data):
    """
    将主题结构的问答对展平成列表，并为每个问答对分配唯一ID
    """
    flat = []
    for topic_group in qa_data:
        topic = topic_group.get('topic', '未知主题')
        for idx, qa in enumerate(topic_group.get('qa_pairs', [])):
            flat.append({
                'topic': topic,
                'question': qa['question'],
                'answer': qa['answer'],
                'id': f"{topic}_{idx}",
            })
    return flat

def is_similar(q1, q2, threshold=0.9):
    """判断两个问答对是否高度相似（同义/重复）"""
    s1 = q1['question'] + q1['answer']
    s2 = q2['question'] + q2['answer']
    return SequenceMatcher(None, s1, s2).ratio() > threshold

def group_similar_qa(qa_list, threshold=0.9):
    """将高度相似的问答对分组，避免拆分到不同集合"""
    groups = []
    used = set()
    for i, qa in enumerate(qa_list):
        if i in used:
            continue
        group = [i]
        for j in range(i+1, len(qa_list)):
            if j in used:
                continue
            if is_similar(qa, qa_list[j], threshold):
                group.append(j)
                used.add(j)
        used.add(i)
        groups.append(group)
    return groups

def stratified_split(qa_list, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1, seed=42):
    """
    按主题分层，分组后随机拆分为train/test/val
    """
    random.seed(seed)
    # 按主题分组
    topic_dict = defaultdict(list)
    for qa in qa_list:
        topic_dict[qa['topic']].append(qa)
    train, test, val = [], [], []
    for topic, qas in topic_dict.items():
        # 分组去重
        groups = group_similar_qa(qas)
        # 打乱分组顺序
        random.shuffle(groups)
        n = len(groups)
        n_train = int(n * train_ratio)
        n_test = int(n * test_ratio)
        # 分配
        for idx, group in enumerate(groups):
            for i in group:
                if idx < n_train:
                    train.append(qas[i])
                elif idx < n_train + n_test:
                    test.append(qas[i])
                else:
                    val.append(qas[i])
    return train, test, val

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            # OpenAI微调格式
            record = {
                'messages': [
                    {"role": "system", "content": f"你是一个专业的供应链技术专家，擅长回答关于{item['topic']}的问题。"},
                    {"role": "user", "content": item['question']},
                    {"role": "assistant", "content": item['answer']}
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='高质量问答对7:2:1分层随机拆分')
    parser.add_argument('--input', required=True, help='高质量问答对json/jsonl')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--train', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--test', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    qa_data = load_qa_data(args.input)
    # 判断是否已是扁平化格式
    if isinstance(qa_data, list) and 'messages' in qa_data[0]:
        print('输入已为微调格式，无需拆分')
        return
    qa_list = flatten_qa_pairs(qa_data)
    # 唯一性校验
    unique = {}
    for qa in qa_list:
        key = qa['question'].strip() + qa['answer'].strip()
        if key not in unique:
            unique[key] = qa
    qa_list = list(unique.values())
    train, test, val = stratified_split(qa_list, args.train, args.test, args.val, args.seed)
    save_jsonl(train, os.path.join(args.output, 'train.jsonl'))
    save_jsonl(test, os.path.join(args.output, 'test.jsonl'))
    save_jsonl(val, os.path.join(args.output, 'val.jsonl'))
    print(f"训练集: {len(train)}，测试集: {len(test)}，验证集: {len(val)}")

if __name__ == "__main__":
    main() 