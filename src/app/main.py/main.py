import streamlit as st 
import tensorflow as tf 
import numpy as np 
from PIL import Image 
import sys 
import os 

# モジュールのパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

#from src.models.classifier import LesionClassifier
from models.classifier import LesionClassifier

def load_model():
    """モデルのロードと初期化"""
    model = LesionClassifier(input_shape=(224, 224, 3), num_classes=2)
    print("Model summary:")
    model.model.summary()
    
    # ルートディレクトリのmodelsフォルダを指すように修正
    current_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    model_path = os.path.join(root_dir, 'models', 'model_with_lr_scheduling_v1.weights.h5')
    
    # デバッグ用にパスを表示
    print(f"Looking for model at: {model_path}")
    
    model.load_weights(model_path)
    return model

def preprocess_image(image):
    """画像の前処理"""
    # PILイメージをnumpy配列に変換
    img_array = np.array(image)
    
    # リサイズ
    img = tf.image.resize(img_array, (224, 224))
    
    # 正規化
    img = img / 225.0
    
    # バッチ次元の追加
    img = np.expand_dims(img, axist=0)
    
    return img

def main():
    st.title("皮膚病変分類アプリケーション")
    
    # サイドバーの設定
    st.sidebar.title("設定")
    
    # メインコンテンツ
    st.write("画像をアップロードして、皮膚病変の分類を行います。")
    
    # モデルのロード
    try:
        model = load_model()
        st.success("モデルのロードが完了しました。")
    except Exception as e:
        st.error(f"モデルのロードに失敗しました。: {e}")
        
        return
    
    # ファイルのアップローダーの配置
    uploaded_file = st.file_uploader("画像を選択してください。", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # 画像の表示
        image = Image.open(uploaded_file)
        st.image(image, caption='アップロードされた画像', use_column_width=True)
        
        # 予測ボタン
        if st.button('分類を実行'):
            # 画像の前処理
            processed_image = preprocess_image(image)
            
            # 予測の実行
            with st.spinner('分類を実行中...'):
                prediction = model.predict(processed_image)
                
                # 結果の表示
                st.subheader("分類結果:")
                class_names = ['良性', '悪性']
                probabilities = prediction[0]
                
                # 結果をプログレスバーで表示
                for name, prob in zip(class_names, probabilities):
                    st.write(f"{name}: {prob:.2%}")
                    st.progress(float(prob))
                    
if __name__ == "__main__":
    main()