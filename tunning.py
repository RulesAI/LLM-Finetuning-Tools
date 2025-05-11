import argparse
from langchain_community.document_loaders import PyPDFLoader        # 基于 pdfminer
from unstructured.cleaners.core import clean
import os

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='处理PDF文件并提取内容')
    parser.add_argument('--pdf', required=True, help='PDF文件路径')
    parser.add_argument('--output', default='processed_content.txt', help='输出文件路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载PDF文件
    loader = PyPDFLoader(args.pdf)
    docs = loader.load()                                     # 每页一个 Document
    pages = [clean(d.page_content) for d in docs]            # 去页眉页脚等杂讯

    # 将处理后的内容保存到文件
    with open(args.output, 'w', encoding='utf-8') as f:
        for i, page_content in enumerate(pages, 1):
            f.write(f'\n\n=== 第 {i} 页 ===\n\n')
            f.write(page_content)
    
    print(f"PDF内容已成功提取并保存到: {args.output}")

if __name__ == "__main__":
    main()