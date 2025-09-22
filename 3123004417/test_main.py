import unittest
import os
import tempfile
import sys
from main import (
    read_file,
    write_result,
    preprocess_text,
    get_word_frequency,
    calculate_cosine_similarity,
    calculate_similarity,
    main
)

class TestPaperChecker(unittest.TestCase):
    """论文查重系统完整单元测试"""
    
    # 测试用例数据
    TEST_CASES = [
        # 正常情况
        ("今天是星期天，天气晴，今天晚上我要去看电影。", 
         "今天是周天，天气晴朗，我晚上要去看电影。", 
         0.82),
        
        # 完全相同
        ("测试文本完全一致", "测试文本完全一致", 1.00),
        
        # 完全不同
        ("这是第一段文本", "这是完全不相关的另一段内容", 0.00),
        
        # 长度差异大
        ("短文本", "这是一段比原文长得多的文本，包含更多的词汇和句子结构", 0.15),
        
        # 包含特殊字符
        ("文本中包含@#$%^等特殊符号", "文本中包含特殊符号", 0.78),
        
        # 中英文混合
        ("Python是一种编程语言，简单易学", 
         "Python is a programming language that's easy to learn", 0.28),
        
        # 近义词测试
        ("这个电影非常好看", "这部影片十分精彩", 0.55)
    ]
    
    def setUp(self):
        """测试前准备：创建临时文件"""
        # 创建临时原文文件
        self.orig_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        self.orig_file.write("测试原文内容")
        self.orig_file.close()
        
        # 创建临时抄袭文件
        self.copy_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        self.copy_file.write("测试抄袭内容")
        self.copy_file.close()
        
        # 创建临时结果文件
        self.result_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        self.result_file.close()
    
    def tearDown(self):
        """测试后清理：删除临时文件"""
        for file in [self.orig_file.name, self.copy_file.name, self.result_file.name]:
            if os.path.exists(file):
                os.unlink(file)
    
    # 测试read_file函数
    def test_read_file_normal(self):
        """测试正常读取文件"""
        content = read_file(self.orig_file.name)
        self.assertEqual(content, "测试原文内容")
    
    def test_read_file_not_exists(self):
        """测试读取不存在的文件"""
        with self.assertRaises(FileNotFoundError):
            read_file("non_existent_file_123.txt")
    
    def test_read_file_is_directory(self):
        """测试路径指向目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(IsADirectoryError):
                read_file(temp_dir)
    
    def test_read_file_empty(self):
        """测试读取空文件"""
        empty_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        empty_file.close()
        with self.assertRaises(ValueError):
            read_file(empty_file.name)
        os.unlink(empty_file.name)
    
    # 测试write_result函数
    def test_write_result_normal(self):
        """测试正常写入结果"""
        write_result(self.result_file.name, 0.85)
        with open(self.result_file.name, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read().strip(), "0.85")
    
    def test_write_result_rounding(self):
        """测试结果四舍五入"""
        write_result(self.result_file.name, 0.856)
        with open(self.result_file.name, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read().strip(), "0.86")
        
        write_result(self.result_file.name, 0.854)
        with open(self.result_file.name, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read().strip(), "0.85")
    
    def test_write_result_boundaries(self):
        """测试边界值写入"""
        write_result(self.result_file.name, 1.0)
        with open(self.result_file.name, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read().strip(), "1.00")
            
        write_result(self.result_file.name, 0.0)
        with open(self.result_file.name, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read().strip(), "0.00")
    
    # 测试preprocess_text函数
    def test_preprocess_text_basic(self):
        """测试文本预处理基础功能"""
        text = "今天是星期天，天气晴，今天晚上我要去看电影。"
        words = preprocess_text(text)
        self.assertIn("今天", words)
        self.assertIn("星期天", words)
        self.assertIn("天气", words)
        self.assertIn("晴", words)
        self.assertIn("晚上", words)
        self.assertIn("看电影", words)
    
    def test_preprocess_text_special_chars(self):
        """测试特殊字符处理"""
        text = "文本@包含#特殊$字符%^&*()"
        words = preprocess_text(text)
        self.assertEqual(words, ["文本", "包含", "特殊", "字符"])
    
    def test_preprocess_text_stopwords(self):
        """测试停用词过滤"""
        text = "的 了 在 是 我 今天 天气"
        words = preprocess_text(text)
        self.assertEqual(words, ["今天", "天气"])
    
    # 测试get_word_frequency函数
    def test_get_word_frequency(self):
        """测试词频计算"""
        words = ["今天", "今天", "天气", "晴", "晚上", "看电影"]
        freq = get_word_frequency(words)
        self.assertEqual(freq["今天"], 2)
        self.assertEqual(freq["天气"], 1)
        self.assertEqual(freq["晴"], 1)
        self.assertEqual(freq["晚上"], 1)
        self.assertEqual(freq["看电影"], 1)
    
    # 测试calculate_cosine_similarity函数
    def test_cosine_similarity_identical(self):
        """测试完全相同的文本相似度"""
        orig_freq = {"a": 2, "b": 3, "c": 1}
        copy_freq = {"a": 2, "b": 3, "c": 1}
        self.assertEqual(calculate_cosine_similarity(orig_freq, copy_freq), 1.0)
    
    def test_cosine_similarity_no_overlap(self):
        """测试无重叠词汇的相似度"""
        orig_freq = {"a": 1, "b": 2}
        copy_freq = {"c": 3, "d": 4}
        self.assertEqual(calculate_cosine_similarity(orig_freq, copy_freq), 0.0)
    
    def test_cosine_similarity_partial_overlap(self):
        """测试部分重叠的相似度"""
        orig_freq = {"a": 1, "b": 2, "c": 3}
        copy_freq = {"b": 1, "c": 2, "d": 3}
        similarity = calculate_cosine_similarity(orig_freq, copy_freq)
        self.assertAlmostEqual(similarity, 0.72, delta=0.01)
    
    def test_cosine_similarity_empty(self):
        """测试空词频的相似度"""
        self.assertEqual(calculate_cosine_similarity({}, {"a": 1}), 0.0)
        self.assertEqual(calculate_cosine_similarity({"a": 1}, {}), 0.0)
        self.assertEqual(calculate_cosine_similarity({}, {}), 0.0)
    
    # 测试calculate_similarity函数
    def test_calculate_similarity_all_cases(self):
        """测试所有预设的相似度计算用例"""
        # 创建临时文件
        orig_temp = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        copy_temp = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        
        try:
            for orig_text, copy_text, expected in self.TEST_CASES:
                with self.subTest(orig=orig_text[:20]):
                    # 写入测试内容
                    orig_temp.seek(0)
                    orig_temp.write(orig_text)
                    orig_temp.flush()
                    
                    copy_temp.seek(0)
                    copy_temp.write(copy_text)
                    copy_temp.flush()
                    
                    # 计算相似度
                    similarity = calculate_similarity(orig_temp.name, copy_temp.name)
                    self.assertAlmostEqual(similarity, expected, delta=0.05)
        finally:
            orig_temp.close()
            copy_temp.close()
            os.unlink(orig_temp.name)
            os.unlink(copy_temp.name)
    
    # 测试main函数
    def test_main_function_normal(self):
        """测试主函数正常执行"""
        # 准备测试数据
        with open(self.orig_file.name, 'w', encoding='utf-8') as f:
            f.write("今天是星期天，天气晴，今天晚上我要去看电影。")
        
        with open(self.copy_file.name, 'w', encoding='utf-8') as f:
            f.write("今天是周天，天气晴朗，我晚上要去看电影。")
        
        # 保存原始命令行参数
        original_argv = sys.argv
        try:
            # 模拟命令行参数
            sys.argv = ['main.py', self.orig_file.name, self.copy_file.name, self.result_file.name]
            main()
            
            # 验证结果
            with open(self.result_file.name, 'r', encoding='utf-8') as f:
                result = f.read().strip()
                self.assertRegex(result, r'^\d+\.\d{2}$')  # 验证格式
                self.assertAlmostEqual(float(result), 0.82, delta=0.05)
        finally:
            # 恢复原始命令行参数
            sys.argv = original_argv
    
    def test_main_function_invalid_args(self):
        """测试主函数参数错误的情况"""
        original_argv = sys.argv
        try:
            # 参数不足
            sys.argv = ['main.py', 'only_one_arg']
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 1)
            
            # 参数过多
            sys.argv = ['main.py', '1', '2', '3', '4']
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 1)
        finally:
            sys.argv = original_argv

if __name__ == '__main__':
    unittest.main(verbosity=2)
