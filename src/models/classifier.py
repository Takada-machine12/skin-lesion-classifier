"""
皮膚病変画像の分類を行うモデルを定義するモジュール
"""
import tensorflow as tf
from keras import layers, Model
from typing import Tuple

class LesionClassifier:
    """皮膚病変の分類を行うモデルクラス"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = [224, 224, 3], num_classes: int = 2, learning_rate: float = 0.001):
        """
        初期化メソッド
        
        Args:
            input_shape (Tuple[int, int, int]): 入力画像のシェイプ(height, width, channels)
            num_classes(int): 分類するクラス数
            learning_rate: 学習率。デフォルトは0.001
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model, self.reduce_lr = self._build_model()
        
    def _residual_block(self, x, filters, kernel_size=3):
        """
        残差ブロックの実装
        
        Args:
            x: 入力テンソル
            filters: フィルタ数
            kernel_size: カーネルサイズ
            
        Returns:
            残差接続を適用した出力テンソル
        """
        # メインパスでの特徴抽出
        y = layers.Conv2D(filters, kernel_size, padding='same')(x)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Conv2D(filters, kernel_size, padding='same')(y)
        y = layers.BatchNormalization()(y)
        
        # 入力と出力のチャネル数が異なる場合の調整
        if x.shape[-1] != filters:
            x = layers.Conv2D(filters, 1)(x)
            
        # 残差接続の適用
        out = layers.Add()([x, y])
        out = layers.ReLU()(out)
        
        return out
        
    def _build_model(self) -> Tuple[Model, tf.keras.callbacks.ReduceLROnPlateau]:
        """
        残差ネットワークベースのCNNモデルの構築
        
        より深い特徴学習を可能にするため、残差ブロックを使用
        勾配消失問題を緩和し、効果的な学習を実現
        
        Returns:
            Tuple[Model, ReduceLROnPlateau]: コンパイル済みのKerasモデルと学習率スケジューラー
        """
        # 入力層の定義
        inputs = layers.Input(shape=self.input_shape)
        
        # 初期の特徴抽出
        x = layers.Conv2D(64, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # 残差ブロックによる深い特徴学習
        x = self._residual_block(x, filters=64)
        x = layers.MaxPooling2D()(x)
        
        x = self._residual_block(x, filters=128)
        x = layers.MaxPooling2D()(x)
        
        x = self._residual_block(x, filters=256)
        x = layers.MaxPooling2D()(x)
        
        # グローバル特徴の抽出
        x = layers.GlobalAveragePooling2D()(x)
        
        # 分類層
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # モデルの構築
        model = Model(inputs, outputs)
        
        # 学習率スケジューラーの設定
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
        
        # オプティマイザと損失関数の設定
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model, reduce_lr
    
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
        
        if callbacks is None:
            callbacks = []
            
        # 学習率スケジューリングのコールバックを追加
        callbacks.append(self.reduce_lr)   
        
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
            filepath: 保存先のパス(.weights.h5で終わる必要がある)
        """
        if not filepath.endswith('.weights.h5'):
            filepath = filepath + '.weights.h5'
        self.model.save_weights(filepath)
        
    def load_weights(self, filepath: str):
        """
        モデルの重みを読み込む
        
        Args:
            filepath: 読み込むファイルのパス
        """
        self.model.load_weights(filepath)
        