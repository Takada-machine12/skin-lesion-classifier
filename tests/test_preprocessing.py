import unittest
import numpy as np
from PIL import Image
import os
from src.app.main import preprocess_image # type: ignore

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """テストの前準備"""
        # テストデータのパス設定
        self.test_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'test')
        self.test_image_path = os.path.join(self.test_dir, 'test_image_0.jpg')
        
    def test_image_loading(self):
        """画像の読み込みテスト"""
        self.assertTrue(os.path.exists(self.test_image_path))
        image = Image.open(self.test_image_path)
        self.assertIsInstance(image, Image.Image)
        
    def test_preprocess_image(self):
        """画像の前処理テスト"""
        image = Image.open(self.test_image_path)
        processed = preprocess_image(image)
        
        # シェイプのテスト
        self.assertEqual(processed.shape[1:], (224, 224, 3))
        # 値の範囲のテスト(0-1の正規化)
        self.assertTrue(0 <= processed.min() <= processed.max() <= 1)
        
    def test_invalid_input(self):
        """無効な入力テスト"""
        with self.assertRaises(Exception):
            preprocess_image(None)

if __name__ == '__main__':
    unittest.main()