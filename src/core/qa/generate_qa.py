import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

#!/usr/bin/env python3
import os
import json
import argparse
import yaml
from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(description='使用OpenAI API生成问答对')
    parser.add_argument('--input', required=True, help='文本段落文件')
    parser.add_argument('--output', required=True, help='问答对输出文件')
    parser.add_argument('--config', default='models_config.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 读取配置
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("成功读取配置文件")
    except Exception as e:
        print(f"读取配置文件出错: {str(e)}")
        return
    
    # 初始化OpenAI客户端
    api_key = config['api_keys']['openai']
    model = config['models']['qa_generation']
    client = OpenAI(api_key=api_key)
    print(f"使用模型: {model}")
    
    # 读取文本段落
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"成功读取文本文件: {args.input}")
    except Exception as e:
        print(f"读取文本文件出错: {str(e)}")
        return
    
    # 分割段落
    segments = []
    current_segment = ""
    in_segment = False
    
    for line in text.split('\n'):
        if line.startswith('=== 段落'):
            if in_segment and current_segment.strip():
                segments.append(current_segment.strip())
            in_segment = True
            current_segment = ""
        elif in_segment:
            current_segment += line + "\n"
    
    # 添加最后一个段落
    if in_segment and current_segment.strip():
        segments.append(current_segment.strip())
    
    print(f"共分割出 {len(segments)} 个段落")
    
    # 为每个段落生成问答对
    all_qa_pairs = []
    topics = []
    
    for i, segment in enumerate(segments):
        print(f"处理段落 {i+1}/{len(segments)}...")
        
        # 检测主题
        topic = "供应链技术"
        for line in segment.split('\n'):
            if line.startswith('##'):
                topic = line.replace('#', '').strip()
                break
        
        if topic not in topics:
            topics.append(topic)
        
        # 构建提示词
        prompt = f"""基于以下供应链技术文档段落生成3个高质量的问答对。每个问答对应该：
1. 问题具体明确，不宜过于开放或笼统
2. 回答准确、全面，直接基于文档内容
3. 回答格式规范，便于学习，可以适当使用编号格式化输出

文档段落:
{segment}

请直接输出以下格式的JSON数组（不需要任何前言和说明）:
[
  {{"question": "问题1", "answer": "回答1"}},
  {{"question": "问题2", "answer": "回答2"}},
  {{"question": "问题3", "answer": "回答3"}}
]"""
        
        try:
            # 调用OpenAI API生成问答对
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的供应链技术知识问答生成器，根据给定的文档内容生成高质量的问答对。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            
            # 获取回复内容
            result = response.choices[0].message.content
            
            # 解析JSON
            try:
                qa_pairs = json.loads(result)
                print(f"成功为段落 {i+1} 生成 {len(qa_pairs)} 个问答对")
                
                # 添加到结果中
                current_topic_found = False
                for t in all_qa_pairs:
                    if t['topic'] == topic:
                        t['qa_pairs'].extend(qa_pairs)
                        current_topic_found = True
                        break
                
                if not current_topic_found:
                    all_qa_pairs.append({
                        'topic': topic,
                        'qa_pairs': qa_pairs
                    })
                
            except json.JSONDecodeError:
                # 尝试从文本中提取JSON
                import re
                json_match = re.search(r'\[\s*\{.*\}\s*\]', result, re.DOTALL)
                if json_match:
                    try:
                        qa_pairs = json.loads(json_match.group(0))
                        print(f"从文本中提取JSON，为段落 {i+1} 生成 {len(qa_pairs)} 个问答对")
                        
                        # 添加到结果中
                        current_topic_found = False
                        for t in all_qa_pairs:
                            if t['topic'] == topic:
                                t['qa_pairs'].extend(qa_pairs)
                                current_topic_found = True
                                break
                        
                        if not current_topic_found:
                            all_qa_pairs.append({
                                'topic': topic,
                                'qa_pairs': qa_pairs
                            })
                    except:
                        print(f"从文本中提取JSON失败，跳过段落 {i+1}")
                else:
                    print(f"无法解析段落 {i+1} 的回复，跳过")
                    
        except Exception as e:
            print(f"调用API出错: {str(e)}")
    
    # 保存结果
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
        print(f"成功保存 {len(all_qa_pairs)} 个主题的问答对到文件: {args.output}")
    except Exception as e:
        print(f"保存结果出错: {str(e)}")

if __name__ == "__main__":
    main() 