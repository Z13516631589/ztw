import sys
import jieba
import math
import os
from collections import defaultdict

# 停用词表
STOPWORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
    '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
    '看', '好', '自己', '这', '个', '中', '为', '以', '于', '等', '能', '对'
}

def read_file(file_path):
    """读取文件内容，处理异常情况"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    if not os.path.isfile(file_path):
        raise IsADirectoryError(f"路径不是文件: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"文件内容为空: {file_path}")
            return content
    except PermissionError:
        raise PermissionError(f"没有权限读取文件: {file_path}")
    except UnicodeDecodeError:
        raise UnicodeDecodeError(f"文件编码错误，无法读取: {file_path}")
    except Exception as e:
        raise Exception(f"读取文件失败: {str(e)}")

def write_result(file_path, similarity):
    """写入结果到文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{similarity:.2f}")
    except PermissionError:
        raise PermissionError(f"没有权限写入文件: {file_path}")
    except Exception as e:
        raise Exception(f"写入文件失败: {str(e)}")

def preprocess_text(text):
    """文本预处理：清洗和分词"""
    # 去除特殊字符
    cleaned = []
    for char in text:
        if char.isalnum() or char.isspace():
            cleaned.append(char)
    text = ''.join(cleaned)
    
    # 分词
    words = jieba.lcut(text)
    
    # 过滤停用词和空字符
    filtered = [word for word in words if word.strip() and word not in STOPWORDS]
    
    return filtered

def get_word_frequency(words):
    """计算词频"""
    freq = defaultdict(int)
    for word in words:
        freq[word] += 1
    return freq

def calculate_cosine_similarity(orig_freq, copy_freq):
    """计算余弦相似度"""
    # 获取所有独特词汇
    all_words = set(orig_freq.keys()).union(set(copy_freq.keys()))
    
    # 计算向量点积和模长
    dot_product = 0
    orig_norm = 0
    copy_norm = 0
    
    for word in all_words:
        o = orig_freq.get(word, 0)
        c = copy_freq.get(word, 0)
        
        dot_product += o * c
        orig_norm += o **2
        copy_norm += c** 2
    
    # 处理模长为0的情况
    if orig_norm == 0 or copy_norm == 0:
        return 0.0
    
    # 计算余弦相似度
    return dot_product / (math.sqrt(orig_norm) * math.sqrt(copy_norm))

def calculate_similarity(orig_path, copy_path):
    """计算两篇论文的相似度"""
    try:
        # 读取文本
        orig_text = read_file(orig_path)
        copy_text = read_file(copy_path)
        
        # 预处理
        orig_words = preprocess_text(orig_text)
        copy_words = preprocess_text(copy_text)
        
        # 计算词频
        orig_freq = get_word_frequency(orig_words)
        copy_freq = get_word_frequency(copy_words)
        
        # 计算相似度
        similarity = calculate_cosine_similarity(orig_freq, copy_freq)
        
        return round(similarity, 4)  # 保留四位小数用于后续四舍五入
    except Exception as e:
        print(f"计算相似度时出错: {str(e)}", file=sys.stderr)
        return 0.0

def main():
    """主函数：处理命令行参数并执行查重"""
    if len(sys.argv) != 4:
        print("用法: python main.py [原文文件路径] [抄袭版文件路径] [结果文件路径]", file=sys.stderr)
        sys.exit(1)
    
    orig_path = sys.argv[1]
    copy_path = sys.argv[2]
    result_path = sys.argv[3]
    
    try:
        similarity = calculate_similarity(orig_path, copy_path)
        write_result(result_path, similarity)
        sys.exit(0)
    except Exception as e:
        print(f"程序执行失败: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()



