import unittest
import numpy as np
from PIL import Image
import os
from src.app.main import preprocess_image, load_model # type: ignore

class TestApp(unittest.TestCase):
    def setUp(self):
        """テストの前準備"""
        self.test_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'test')
        self.test_image_benign = os.path.join(self.test_dir, 'test_image_0.jpg')
        self.test_image_malignant = os.path.join(self.test_dir, 'test_image_1.jpg')
        self.model = load_model()
        
    def test_end_to_end_prediction_benign(self):
        """良性画像の予測テスト"""
        # 画像の読み込みと前処理
        image = Image.open(self.test_image_benign)
        processed_image = preprocess_image(image)
        
        # 予測
        predictions = self.model.predict(processed_image)
        
        # 予測結果の検証(良性の確率が高いことを確認)
        self.assertTrue(predictions[0][0] > 0.5) # クラス0(良性)の確率が50%以上
        
    def test_end_to_end_prediction_malignant(self):
        """悪性画像の予測テスト"""
        # 画像の読み込みと前処理
        image = Image.open(self.test_image_malignant)
        processed_image = preprocess_image(image)
        
        # 予測
        predictions = self.model.predict(processed_image)
        
        # 予測結果の検証(悪性の確率が高いことを確認)
        self.assertTrue(predictions[0][1] > 0.5) # クラス1(悪性)の確率が50%以上
        
    def test_invalid_file_type(self):
        """無効なファイル形式のテスト"""
        invalid_file = os.path.join(self.test_dir, 'test.txt')
        with open(invalid_file, 'w') as f:
            f.write('test')
            
        # 例外が発生することを確認
        with self.assertRaises(Exception):
            with opne(invalid_file, 'rb') as f:
                preprocess_image(f)
                
if __name__ == '__main__':
    unittest.main()