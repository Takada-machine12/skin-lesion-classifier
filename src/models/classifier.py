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
        
        # オプティマイザーの学習率を調整
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) # 0.001 → 0.0001
        
        # モデルのコンパイル
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self,
              x_train, 
              y_train,
              validation_data=None,
              batch_size=32,
              epochs=10,
              callbacks=None,
              steps_per_epoch=None) -> tf.keras.callbacks.History:
        """
        モデルの学習を行う
        
        Args:
            X_train: 訓練用データ
            y_train: 検証用データ
            validation_data: 検証データのタプル(X_val, y_val)
            batch_size: バッチサイズ
            epochs: 学習エポック数
            callbacks: コールバック関数のリスト
            steps_per_epoch: １エポックあたりのステップ数(データ拡張時に使用)
            
        Returns:
            History: 学習履歴
        """
        return self.model.fit(
            x_train, 
            y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch # fitメソッドにも追加
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
    
    def save_weights(self, filepath: str):
        """
        モデルの重みを保存する
        
        Args:
            filepath: 保存先のパス
        """
        self.model.save_weights(filepath)
        
    def load_weights(self, filepath: str):
        """
        モデルの重みを読み込む
        
        Args:
            filepath: 読み込むファイルのパス
        """
        self.model.load_weights(filepath)
        