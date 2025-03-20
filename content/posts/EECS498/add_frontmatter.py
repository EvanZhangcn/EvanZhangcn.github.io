import os
from datetime import datetime

def create_front_matter(title, date):
    """创建前置元数据，包含url字段"""
    return f"""---
title: "{title}"
date: "2025-03-20 09:49:07"
author: "EvanZhangcn"
draft: false
categories: ["未分类"]  # 在此编辑分类
tags: []               # 在此添加标签
url: "/posts/EECS498/{title}"  # 自动生成的URL
---

"""

def process_markdown_file(file_path):
    """处理单个Markdown文件"""
    try:
        # 从文件名获取标题（移除.md后缀）
        title = os.path.splitext(os.path.basename(file_path))[0]
        
        # 读取原始文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 检查文件是否已经有前置元数据
        if content.startswith('---'):
            print(f"文件 {file_path} 已有前置元数据，跳过处理")
            return
        
        # 创建新的前置元数据
        front_matter = create_front_matter(title, "2025-03-20 09:49:07")
        
        # 将前置元数据添加到文件开头
        new_content = front_matter + content
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        
        print(f"已处理文件: {file_path}")
        print(f"生成的URL: /posts/EECS498/{title}")
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")

def process_directory(directory):
    """处理目录中的所有Markdown文件"""
    try:
        md_files = [f for f in os.listdir(directory) if f.endswith('.md')]
        if not md_files:
            print("当前目录下没有找到Markdown文件！")
            return False
        
        print(f"找到 {len(md_files)} 个Markdown文件")
        for md_file in md_files:
            file_path = os.path.join(directory, md_file)
            process_markdown_file(file_path)
            
        return True
    
    except Exception as e:
        print(f"处理目录时出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 使用当前目录
    current_directory = os.getcwd()
    print(f"正在处理目录: {current_directory}")
    
    if os.path.exists(current_directory):
        if process_directory(current_directory):
            print("所有Markdown文件处理完成！")
        else:
            print("处理过程中出现错误！")
    else:
        print(f"目录 {current_directory} 不存在！")