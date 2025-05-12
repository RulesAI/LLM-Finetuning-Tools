import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

#!/usr/bin/env python3
import json
import os
import argparse
from src.core.utils.model_utils import ModelCaller

def main():
    parser = argparse.ArgumentParser(description='从分段内容生成问答对')
    parser.add_argument('--input', required=True, help='分段内容JSON文件路径')
    parser.add_argument('--output', required=True, help='输出问答对JSON文件路径')
    parser.add_argument('--config', default='models_config.yaml', help='模型配置文件')
    args = parser.parse_args()
    
    # 初始化模型调用器
    model_caller = ModelCaller(config_path=args.config)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 读取分段内容
        with open(args.input, 'r', encoding='utf-8') as f:
            content = f.read()
            segments = json.loads(content)
        print(f"已成功读取分段文件: {args.input}, 共{len(segments)}个段落")
    except Exception as e:
        print(f"读取分段文件出错: {str(e)}")
        return
    
    qa_data = []
    current_topic = "供应链技术"
    qa_pairs = []
    
    # 处理每个段落生成问答对
    for segment in segments:
        segment_content = segment.get('content', '')
        if not segment_content:
            continue
        
        # 检测段落是否包含新主题
        if segment_content.startswith('##'):
            # 如果已有问答对，保存前一个主题
            if qa_pairs:
                qa_data.append({
                    "topic": current_topic,
                    "qa_pairs": qa_pairs
                })
                qa_pairs = []
            
            # 提取新主题
            lines = segment_content.split('\n')
            for line in lines:
                if line.startswith('##'):
                    current_topic = line.replace('#', '').strip()
                    break
        
        # 为当前段落生成问答对
        prompt = f"""
基于以下供应链技术文档段落生成3个高质量的问答对。每个问答对应该：
1. 问题具体明确，不宜过于开放或笼统
2. 回答准确、全面，直接基于文档内容
3. 回答格式规范，便于学习，可以适当使用编号格式化输出

文档段落:
{segment_content}

请直接输出以下格式的JSON数组（不需要任何前言和说明）:
[
  {{"question": "问题1", "answer": "回答1"}},
  {{"question": "问题2", "answer": "回答2"}},
  {{"question": "问题3", "answer": "回答3"}}
]
"""
        try:
            # 调用模型生成问答对
            result = model_caller.call_model('qa_generation', prompt)
            
            # 尝试解析JSON结果
            try:
                # 尝试直接解析
                new_qa_pairs = json.loads(result)
                print(f"成功为段落 {segment['segment_id']} 生成 {len(new_qa_pairs)} 个问答对")
            except json.JSONDecodeError:
                # 尝试从文本中提取JSON部分
                import re
                json_match = re.search(r'\[\s*\{.*\}\s*\]', result, re.DOTALL)
                if json_match:
                    try:
                        new_qa_pairs = json.loads(json_match.group(0))
                        print(f"成功从文本中提取并解析JSON，为段落 {segment['segment_id']} 生成 {len(new_qa_pairs)} 个问答对")
                    except json.JSONDecodeError:
                        print(f"解析段落 {segment['segment_id']} 的问答结果失败")
                        continue
                else:
                    print(f"无法在段落 {segment['segment_id']} 的模型返回结果中找到JSON数组")
                    continue
            
            # 添加到当前主题的问答对
            qa_pairs.extend(new_qa_pairs)
            
        except Exception as e:
            print(f"处理段落 {segment['segment_id']} 出错: {str(e)}")
    
    # 添加最后一个主题
    if qa_pairs:
        qa_data.append({
            "topic": current_topic,
            "qa_pairs": qa_pairs
        })
    
    # 保存结果
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, ensure_ascii=False, indent=2)
        print(f"问答对已保存到: {args.output}, 共{len(qa_data)}个主题")
    except Exception as e:
        print(f"保存问答结果出错: {str(e)}")

if __name__ == "__main__":
    main() 