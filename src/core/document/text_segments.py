import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

#!/usr/bin/env python3
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='将JSON分段内容转换为纯文本')
    parser.add_argument('--input', required=True, help='JSON分段文件')
    parser.add_argument('--output', required=True, help='输出纯文本文件')
    args = parser.parse_args()
    
    # 读取JSON文件
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取JSON文件: {args.input}")
    except Exception as e:
        print(f"读取JSON文件出错: {str(e)}")
        return
    
    # 提取文本内容
    segments_text = []
    for segment in data:
        if 'content' in segment:
            segments_text.append(segment['content'])
    
    # 将文本内容写入文件
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            for i, text in enumerate(segments_text):
                f.write(f"=== 段落 {i+1} ===\n\n")
                f.write(text + "\n\n")
        print(f"已将{len(segments_text)}个段落内容写入文件: {args.output}")
    except Exception as e:
        print(f"写入文件出错: {str(e)}")
        return

if __name__ == "__main__":
    main() 