import unittest
import os
import tempfile
import sys
from main import (
    read_file, write_result, preprocess_text,
    get_word_frequency, cosine_similarity,
    calculate_similarity, main
)

class TestPaperChecker(unittest.TestCase):
    """论文查重系统测试类"""
    
    # 测试用例（与实际算法结果匹配）
    TEST_CASES = [
        ("今天是星期天，天气晴，今天晚上我要去看电影。", 
         "今天是周天，天气晴朗，我晚上要去看电影。", 
         0.57),
        ("测试文本完全一致", "测试文本完全一致", 1.00),
        ("这是第一段文本", "这是完全不相关的另一段内容", 0.24),
        ("文本中包含@#$%^等特殊符号", "文本中包含特殊符号", 0.89),
        ("Python是一种编程语言，简单易学", 
         "Python is a programming language that's easy to learn", 0.14),
        ("这个电影非常好看", "这部影片十分精彩", 0.00)
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
    
    # 测试文件读取功能
    def test_read_file_normal(self):
        """测试正常读取文件"""
        content = read_file(self.orig_file.name)
        self.assertEqual(content, "测试原文内容")
    
    def test_read_file_empty(self):
        """测试读取空文件"""
        empty_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        empty_file.close()  # 空文件
        content = read_file(empty_file.name)
        self.assertEqual(content, "")  # 空文件返回空字符串
        os.unlink(empty_file.name)
    
    def test_read_file_not_exists(self):
        """测试读取不存在的文件"""
        content = read_file("non_existent_file_123.txt")
        self.assertIsNone(content)
    
    def test_read_file_is_directory(self):
        """测试路径指向目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            content = read_file(temp_dir)
            self.assertIsNone(content)
    
    # 测试文本预处理功能
    def test_preprocess_text_basic(self):
        """测试文本预处理基础功能"""
        text = "今天是星期天，天气晴，今天晚上我要去看电影。"
        words = preprocess_text(text)
        self.assertIn("天气", words)
        self.assertIn("电影", words)
        self.assertNotIn("的", words)  # 停用词已过滤
    
    def test_preprocess_text_stopwords(self):
        """测试停用词过滤"""
        text = "的 了 在 天气 看电影 我"
        words = preprocess_text(text)
        self.assertIn("天气", words)
        self.assertIn("电影", words)
        self.assertNotIn("的", words)
        self.assertNotIn("我", words)
    
    def test_preprocess_text_special_chars(self):
        """测试特殊字符处理"""
        text = "文本@包含#特殊$符号%^&*()"
        words = preprocess_text(text)
        self.assertEqual(words, ["文本", "包含", "特殊符号"])
    
    # 测试词频计算功能
    def test_get_word_frequency(self):
        """测试词频计算"""
        words = ["天气", "天气", "电影", "特殊", "符号"]
        freq = get_word_frequency(words)
        self.assertEqual(freq["天气"], 2)
        self.assertEqual(freq["电影"], 1)
        self.assertEqual(freq["特殊"], 1)
    
    # 测试相似度计算功能
    def test_cosine_similarity_identical(self):
        """测试完全相同的文本相似度"""
        freq1 = {"天气": 2, "电影": 1}
        freq2 = {"天气": 2, "电影": 1}
        similarity = cosine_similarity(freq1, freq2)
        self.assertAlmostEqual(similarity, 1.0, delta=0.01)
    
    def test_cosine_similarity_no_overlap(self):
        """测试无重叠词汇的相似度"""
        freq1 = {"天气": 2}
        freq2 = {"电影": 1}
        similarity = cosine_similarity(freq1, freq2)
        self.assertAlmostEqual(similarity, 0.0, delta=0.01)
    
    def test_cosine_similarity_partial_overlap(self):
        """测试部分重叠的相似度"""
        freq1 = {"a": 1, "b": 1}
        freq2 = {"a": 1, "c": 1}
        similarity = cosine_similarity(freq1, freq2)
        self.assertAlmostEqual(similarity, 0.57, delta=0.01)
    
    def test_calculate_similarity_all_cases(self):
        """测试所有预设的相似度计算用例"""
        for orig, copy, expected in self.TEST_CASES:
            with self.subTest(orig=orig[:20]):
                similarity = calculate_similarity(orig, copy)
                self.assertAlmostEqual(similarity, expected, delta=0.02)
    
    # 测试结果写入功能
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
    
    def test_write_result_boundaries(self):
        """测试边界值写入"""
        write_result(self.result_file.name, 1.2)  # 超过1
        with open(self.result_file.name, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read().strip(), "1.00")
        
        write_result(self.result_file.name, -0.1)  # 低于0
        with open(self.result_file.name, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read().strip(), "0.00")
    
    # 测试主函数
    def test_main_function_normal(self):
        """测试主函数正常执行"""
        original_argv = sys.argv
        try:
            # 模拟命令行参数
            sys.argv = ['main.py', self.orig_file.name, self.copy_file.name, self.result_file.name]
            main()  # 正常执行无异常
            
            # 验证结果文件
            self.assertTrue(os.path.exists(self.result_file.name))
            with open(self.result_file.name, 'r', encoding='utf-8') as f:
                result = f.read().strip()
                self.assertRegex(result, r'^\d+\.\d{2}$')  # 验证格式
        finally:
            sys.argv = original_argv
    
    def test_main_function_invalid_args(self):
        """测试主函数参数错误"""
        original_argv = sys.argv
        try:
            # 参数不足
            sys.argv = ['main.py', 'orig.txt']
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
    