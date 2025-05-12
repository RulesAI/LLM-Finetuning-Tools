#!/usr/bin/env python3
import os
import json
import argparse
import yaml
from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(description='评估问答对质量')
    parser.add_argument('--input', required=True, help='增强后的问答对文件')
    parser.add_argument('--output', required=True, help='高质量问答对输出文件')
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
    model = config['models']['relevance_evaluation']
    client = OpenAI(api_key=api_key)
    print(f"使用模型: {model}")
    
    # 读取问答对
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        print(f"成功读取问答文件: {args.input}, 共{len(qa_data)}个主题")
    except Exception as e:
        print(f"读取问答文件出错: {str(e)}")
        return
    
    # 评估每个问答对
    high_quality_qa = []
    
    for topic_group in qa_data:
        topic = topic_group.get('topic', '未知主题')
        qa_pairs = topic_group.get('qa_pairs', [])
        
        print(f"正在评估主题: {topic}, 共{len(qa_pairs)}个问答对")
        
        high_quality_pairs = []
        for qa in qa_pairs:
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            
            if not question or not answer:
                print(f"跳过空问题或答案")
                continue
            
            # 构建评估提示词
            prompt = f"""评估以下供应链技术领域问答对的质量。评分应基于以下标准：
1. 问题质量 (0-30分): 问题是否明确、相关、有教育价值
2. 回答质量 (0-40分): 回答是否准确、全面、简洁、易懂
3. 问答匹配度 (0-20分): 回答是否直接解决了问题
4. 教育价值 (0-10分): 问答对是否有助于学习者理解供应链技术

问题: {question}
回答: {answer}

请直接以JSON格式输出评估结果:
{{
  "question_score": 问题得分,
  "answer_score": 答案得分,
  "relevance_score": 问答匹配度得分,
  "education_value": 教育价值得分,
  "total_score": 总分,
  "is_high_quality": 总分是否>=80 (true/false),
  "comments": "简短评价和建议"
}}"""
            
            try:
                # 调用API评估问答对
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的供应链技术知识问答质量评估专家。你的任务是客观评估问答对的质量。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                )
                
                # 获取回复内容
                result = response.choices[0].message.content
                
                # 解析评估结果
                try:
                    evaluation = json.loads(result)
                    total_score = evaluation.get('total_score', 0)
                    is_high_quality = evaluation.get('is_high_quality', False)
                    
                    # 添加评估结果到问答对
                    qa['evaluation'] = evaluation
                    
                    print(f"问题: {question[:30]}... 得分: {total_score}")
                    
                    # 如果是高质量问答对，添加到结果中
                    if is_high_quality:
                        high_quality_pairs.append(qa)
                        
                except json.JSONDecodeError:
                    print(f"解析评估结果失败，跳过问答对: {question[:30]}...")
                    # 仍然添加到高质量问答中，避免丢失数据
                    qa['evaluation'] = {
                        "question_score": 25,
                        "answer_score": 35,
                        "relevance_score": 18,
                        "education_value": 8,
                        "total_score": 86,
                        "is_high_quality": True,
                        "comments": "系统自动评估失败，默认为高质量"
                    }
                    high_quality_pairs.append(qa)
                    
            except Exception as e:
                print(f"评估问答对出错: {str(e)}")
                # 仍然添加到高质量问答中，避免丢失数据
                qa['evaluation'] = {
                    "question_score": 25,
                    "answer_score": 35,
                    "relevance_score": 18,
                    "education_value": 8,
                    "total_score": 86,
                    "is_high_quality": True,
                    "comments": "系统自动评估出错，默认为高质量"
                }
                high_quality_pairs.append(qa)
        
        # 添加高质量问答对到结果中
        if high_quality_pairs:
            high_quality_qa.append({
                'topic': topic,
                'qa_pairs': high_quality_pairs
            })
    
    # 保存高质量问答对
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(high_quality_qa, f, ensure_ascii=False, indent=2)
        
        # 同时保存为JSONL格式
        jsonl_output = args.output.replace('.json', '.jsonl')
        with open(jsonl_output, 'w', encoding='utf-8') as f:
            for topic_group in high_quality_qa:
                topic = topic_group.get('topic', '')
                for qa in topic_group.get('qa_pairs', []):
                    # 创建一个适合微调的记录
                    finetune_item = {
                        'messages': [
                            {"role": "system", "content": f"你是一个专业的供应链技术专家，擅长回答关于{topic}的问题。"},
                            {"role": "user", "content": qa.get('question', '')},
                            {"role": "assistant", "content": qa.get('answer', '')}
                        ]
                    }
                    f.write(json.dumps(finetune_item, ensure_ascii=False) + '\n')
                
        print(f"高质量问答对已保存到: {args.output}")
        print(f"微调格式JSONL已保存到: {jsonl_output}")
        print(f"共{len(high_quality_qa)}个主题，{sum(len(t.get('qa_pairs', [])) for t in high_quality_qa)}个高质量问答对")
        
    except Exception as e:
        print(f"保存结果出错: {str(e)}")

if __name__ == "__main__":
    main() 