import json
import re
import argparse
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='对文本内容进行语义分段')
    parser.add_argument('--input', required=True, help='输入文件路径')
    parser.add_argument('--output', default='segmented_content.json', help='输出文件路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 读取处理后的内容
    with open(args.input, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 创建分词器
    splitter = RecursiveCharacterTextSplitter(
        # 分隔符，按优先级排序
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        # 块大小（字符数）
        chunk_size=800,
        # 块重叠（字符数）
        chunk_overlap=150,
        # 长度函数
        length_function=len,
    )
    
    # 按结构元素分段
    # 先去除页码标记
    cleaned_content = re.sub(r'===\s*第\s*\d+\s*页\s*===', '', content)
    chunks = splitter.create_documents([cleaned_content])
    
    # 处理分割结果，添加索引和清理内容
    segments = []
    for i, chunk in enumerate(chunks):
        # 移除过多的空白字符
        segment_text = re.sub(r'\s+', ' ', chunk.page_content).strip()
        if len(segment_text) > 50:  # 过滤太短的段落
            segments.append({
                "segment_id": i,
                "content": segment_text
            })
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存分段结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    
    print(f"文本已成功分段，共 {len(segments)} 个段落，结果已保存到 {args.output}")

if __name__ == "__main__":
    main() 