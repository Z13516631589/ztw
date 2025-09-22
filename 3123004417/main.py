import sys
import os
import re
import jieba
import math

# 停用词表（仅过滤无意义虚词）
STOPWORDS = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', 
             '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', 
             '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '们', '来', '去'}

def read_file(file_path):
    """读取文件内容，空文件返回空字符串，错误返回None"""
    try:
        if os.path.isdir(file_path):
            print(f"错误：{file_path} 是目录，不是文件")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return content  # 空文件返回空字符串（不抛异常）
    
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return None
    except PermissionError:
        print(f"错误：没有读取 {file_path} 的权限")
        return None
    except Exception as e:
        print(f"读取文件错误：{str(e)}")
        return None


def write_result(result_path, similarity):
    """写入结果，处理边界值"""
    try:
        # 确保相似度在0-1范围内
        similarity = max(0.0, min(1.0, similarity))
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f"{similarity:.2f}")
        return True
    except Exception as e:
        print(f"写入结果错误：{str(e)}")
        return False


def preprocess_text(text):
    """文本清洗+分词+停用词过滤"""
    if not text:
        return []
    
    # 清洗特殊字符（保留中文、英文、数字和空格）
    text_clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    # 分词（使用精确模式）
    words = jieba.cut(text_clean, cut_all=False)
    # 过滤停用词和短词
    filtered_words = [
        word for word in words 
        if word.strip() and word not in STOPWORDS and len(word) > 1
    ]
    return filtered_words


def get_word_frequency(words):
    """计算词频"""
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return freq


def cosine_similarity(freq1, freq2):
    """计算余弦相似度"""
    # 获取所有独特词汇
    all_words = set(freq1.keys()).union(set(freq2.keys()))
    if not all_words:
        return 0.0
    
    # 计算点积和模长
    dot_product = 0
    norm1 = 0
    norm2 = 0
    
    for word in all_words:
        count1 = freq1.get(word, 0)
        count2 = freq2.get(word, 0)
        dot_product += count1 * count2
        norm1 += count1 **2
        norm2 += count2** 2
    
    # 避免除以0
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))


def calculate_similarity(orig_text, copy_text):
    """计算文本相似度"""
    try:
        # 预处理文本
        orig_words = preprocess_text(orig_text)
        copy_words = preprocess_text(copy_text)
        
        # 计算词频
        orig_freq = get_word_frequency(orig_words)
        copy_freq = get_word_frequency(copy_words)
        
        # 计算余弦相似度
        return cosine_similarity(orig_freq, copy_freq)
    except Exception as e:
        print(f"计算相似度时出错：{str(e)}")
        return 0.0


def main():
    """主函数：处理命令行参数并执行查重流程"""
    # 校验命令行参数
    if len(sys.argv) != 4:
        print("用法: python main.py [原文文件路径] [抄袭版文件路径] [结果文件路径]")
        sys.exit(1)
    
    # 解析参数
    orig_path, copy_path, result_path = sys.argv[1], sys.argv[2], sys.argv[3]
    
    # 读取文件内容
    orig_text = read_file(orig_path)
    copy_text = read_file(copy_path)
    
    # 检查文件读取是否成功
    if orig_text is None or copy_text is None:
        sys.exit(1)
    
    # 计算相似度
    similarity = calculate_similarity(orig_text, copy_text)
    
    # 写入结果
    if not write_result(result_path, similarity):
        sys.exit(1)
    
    # 成功提示
    print(f"相似度计算完成，结果已保存到 {result_path}（相似度：{similarity:.2f}）")


if __name__ == "__main__":
    main()
    