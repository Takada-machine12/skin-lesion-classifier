import unittest
import numpy as np
from PIL import Image
import os
from src.app.main import load_model # type: ignore

class TestModel(unittest.TestCase):
    def setUp(self):
        """テストの前準備"""
        self.test_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'test')
        self.test_image_benign = os.path.join(self.test_dir, 'test_image_0.jpg') # 良性画像
        self.test_image_malignant = os.path.join(self.test_dir, 'test_image_1.jpg') # 悪性画像
        self.model = load_model()
        
    def test_model_loading(self):
        """モデルのロードテスト"""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.model) # モデルインスタンスの確認
        
    def test_prediction_shape(self):
        """予測のシェイプテスト"""
        image = Image.open(self.test_image_benign)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = self.model.predict(img_array)
        
        # 予測結果は[バッチサイズ、クラス数]の形状
        self.assertEqual(len(predictions.shape), 2)
        self.assertEqual(predictions.shape[1], 2) # 2クラス(良性/悪性)
        
    def test_prediction_range(self):
        """予測値の範囲テスト"""
        image = Image.opne(self.test_image_benign)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = self.model.predict(img_array)
        
        # 確率値は0-1の範囲
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))
        # 確率の合計が1に近いことを確認
        self.assertTrue(np.abs(np.sum(predictions[0]) - 1.0) < 1e-6)
        
if __name__ == '__main__':
    unittest.main()