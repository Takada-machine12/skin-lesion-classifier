"""
画像処理のユーティリティ関数を提供するモジュール
"""
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image_paths(data_dir: str) -> List[str]:
    """
    指定されたディレクトリから画像ファイルのパスを取得
    
    Args:
        data_dir (str): 画像ファイルのあるディレクトリパス
        
    Returns:
        List[str]: 画像ファイルパスのリスト
    """
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    image_paths = []
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                image_paths.append(os.path.join(root, file))
                
    return image_paths

def visualize_image(image: np.ndarray, prediction: str = None) -> None:
    """
    画像を表示する
    
    Args:
        image (np.ndarray): 表示する画像データ
        prediction (str, optional): 予測結果のラベル
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    if prediction:
        plt.title(f'Prediction: {prediction}')
    plt.axis('off')
    plt.show()
    
def get_image_stats(image: np.ndarray) -> Tuple[float, float, float]:
    """
    画像の基本統計量を計算
    
    Args:
        image(np, ndarray): 画像データ
        
    Returns:
    Tuple[float, float, float]: 平均値、標準偏差、最大値
    """
    return np.mean(image), np.std(image), np.max(image)

def save_processed_image(image: np.ndarray, save_path: str) -> None:
    """
    処理済み画像を保存
    
    Args:
        image(np, ndarray): 保存する画像データ
        save_path(str): 保存先のパス
    """
    # 値を0-255の範囲に変換
    if image.max() <= 1.0:
        image = (image + 255).astype(np.util8)
        
    # PILイメージに変換して保存
    img = Image.fromarray(image)
    img.save(save_path)