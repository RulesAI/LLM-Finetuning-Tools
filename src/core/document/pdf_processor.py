import sys
import os
import argparse
import logging
import fitz  # PyMuPDF

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/document/pdf_processor.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pdf_processor")

def extract_text_from_pdf(pdf_path, output_path):
    """
    从PDF中提取文本
    
    参数:
    pdf_path: PDF文件路径
    output_path: 输出文本文件路径
    
    返回:
    bool: 是否成功
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 打开PDF文件
        logger.info(f"正在打开PDF文件: {pdf_path}")
        doc = fitz.open(pdf_path)
        logger.info(f"PDF文件打开成功，共{len(doc)}页")
        
        # 提取文本
        extracted_text = ""
        for page_num, page in enumerate(doc):
            text = page.get_text()
            extracted_text += text
            logger.info(f"已提取第{page_num+1}页文本，长度: {len(text)}字符")
        
        # 保存提取的文本
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        logger.info(f"文本提取完成，已保存到: {output_path}")
        logger.info(f"总文本长度: {len(extracted_text)}字符")
        
        return True
    except Exception as e:
        logger.error(f"提取PDF文本时出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='从PDF中提取文本')
    parser.add_argument('--input', required=True, help='输入PDF文件路径')
    parser.add_argument('--output', required=True, help='输出文本文件路径')
    
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs('logs/document', exist_ok=True)
    
    # 提取文本
    success = extract_text_from_pdf(args.input, args.output)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 