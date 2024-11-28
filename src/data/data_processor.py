"""
皮膚病変画像の前処理を行うモジュール
"""
import os
from typing import Tuple

import numpy as np
from PIL import Image
import tensorflow as tf 

class DataProcessor:
    """画像データの前処理を行うクラス"""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        """
        初期化メソッド
        
        Args:
            image_size (Tuple[int, int]): 画像のリサイズサイズ(height, width)
        """
        self.image_size = image_size
        
    def process_image(self, image_path: str) -> np.ndarray:
        """
        画像の前処理を行う
        
        Args:
            image_path(str): 画像ファイルのパス
            
        Returns:
            np.ndarray: 前処理済みの画像データ
        """
        # 画像の読み込み
        image = image.open(image_path)
        
        # RGBに変換(グレースケールやRGBA画像に対応するため)
        image = image.convert('RGB')
        
        # リサイズ
        image = image.resize(self.image_size)
        
        # numpy配列に変換
        image_array = np.array(image)
        
        # 正規化(0-255 -> 0-1)
        image_array = image_array.astype(np.float32) / 255.0
        
        return image_array
    
    def prepare_dataset(self, data_dir: str) -> tf.data.Dataset:
        """
        データセットの準備
        
        Args:
            data_dir (str): データディレクトリのパス
            
        Returns:
            tf.data.Dataset: 前処理済みのデータセット
        """
        # 実装予定
        pass