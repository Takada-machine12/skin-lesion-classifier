"""
皮膚病変画像の分類を行うモデルを定義するモジュール
"""
import tensorflow as tf
from keras import layers, Model
from typing import Tuple

class LesionClassifier:
    """皮膚病変の分類を行うモデルクラス"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = [224, 224, 3], num_classes: int = 2):
        """
        初期化メソッド
        
        Args:
            input_shape (Tuple[int, int, int]): 入力画像のシェイプ(height, width, channels)
            num_classes(int): 分類するクラス数
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """
        CNNモデルの構築
        
        Returns:
            Model: コンパイル済みのKerasモデル
        """
        model = tf.keras.Sequential([
            # 入力層
            layers.Input(shape=self.input_shape),
            
            # 第１畳込みブロック
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # 第2畳込みブロック
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # 第3畳込みブロック
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # 全結合層への変換
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            
            # 出力層
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # モデルのコンパイル
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self,
              train_dataset: tf.data.Dataset,
              validation_dataset: tf.data.Dataset,
              epochs: int = 10,
              callbacks: list = None) -> tf.keras.callbacks.History:
        """
        モデルの学習を行う
        
        Args:
            train_dataset: 学習用データセット
            validation_dataset: 検証用データセット
            epochs: 学習エポック数
            callbacks: コールバック関数のリスト
        """
        return self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
    def predict(self, image: tf.Tensor) -> tf.Tensor:
        """
        画像の分類を行う
        
        Args:
            image: 入力画像
        
        Returns:
            予測結果
        """
        return self.model.predict(image)
        