import os
import re
import shutil
from pathlib import Path
import urllib.parse

def convert_image_links(content):
    # 1. 处理带有 title 属性和可能包含空格的 Markdown 图片
    md_with_title_pattern = r'!\[(.*?)\]\(((?:\.\.)?\/attachments\/[^"\)]+?)(?:\s+"([^"]+)")?\)'
    
    # 2. 匹配普通的 Markdown 格式图片链接
    md_pattern = r'!\[(.*?)\](\./attachments/[^"\)]+\))'
    
    # 3. 匹配 HTML <img> 标签格式
    html_pattern = r'<img\s+src="(\.\/attachments\/[^"]+)"\s+(?:width="(\d+)"\s+height="(\d+)"\s+)?alt="([^"]+)">'
    
    def replace_md_with_title(match):
        alt_text = match.group(1)
        file_path = match.group(2)
        title = match.group(3)  # 尺寸信息，如 "600x328"
        
        # 处理文件名中的空格，将空格编码为 %20
        filename = os.path.basename(file_path)
        encoded_filename = urllib.parse.quote(filename)
        
        if title:
            return f'![{alt_text}](../attachments/{encoded_filename} "{title}")'
        else:
            return f'![{alt_text}](../attachments/{encoded_filename})'
    
    def replace_md_link(match):
        alt_text = match.group(1)
        file_path = match.group(2)
        filename = os.path.basename(file_path.rstrip(')'))
        encoded_filename = urllib.parse.quote(filename)
        return f'![{alt_text}](../attachments/{encoded_filename})'
    
    def replace_html_link(match):
        file_path = match.group(1)
        width = match.group(2)
        height = match.group(3)
        alt_text = match.group(4)
        filename = os.path.basename(file_path)
        encoded_filename = urllib.parse.quote(filename)
        
        if width and height:
            return f'![{alt_text}](../attachments/{encoded_filename} "{width}x{height}")'
        else:
            return f'![{alt_text}](../attachments/{encoded_filename})'
    
    # 按顺序处理所有格式
    content = re.sub(md_with_title_pattern, replace_md_with_title, content)
    content = re.sub(md_pattern, replace_md_link, content)
    content = re.sub(html_pattern, replace_html_link, content)
    
    return content

def process_md_files(directory):
    # 获取目录下所有的 .md 文件
    md_files = Path(directory).glob('*.md')
    
    for md_file in md_files:
        print(f"Processing {md_file.name}...")
        
        # 读取原文件内容
        with open(md_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 转换内容
        new_content = convert_image_links(content)
        
        # 如果内容有变化，写回文件
        if new_content != content:
            # 创建备份
            backup_file = md_file.with_suffix('.md.bak')
            shutil.copy2(md_file, backup_file)
            
            # 写入新内容
            with open(md_file, 'w', encoding='utf-8') as file:
                file.write(new_content)
            print(f"✓ Converted {md_file.name} (backup created)")
        else:
            print(f"- No changes needed for {md_file.name}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("Starting conversion process...")
    print(f"Working directory: {current_dir}")
    process_md_files(current_dir)
    print("\nConversion complete!")