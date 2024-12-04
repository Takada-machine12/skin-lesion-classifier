# Skin Lesion Classifier
皮膚病変の画像分類を行うディープラーニングベースの分類システム

## 概要

このプロジェクトは、医療画像（皮膚病変）の自動分類システムを実装します。
TensorFlowを使用して、２種類の皮膚病変を分類することができます。
また初めての開発なので、生成AIをフル活用しながら開発を進めております。ご了承ください。

## 必要要件

- Python 3.11
- TensorFlow 2.15.0
- その他の依存関係は `requirements.txt`に記載

## セットアップ

```bash
# 仮想環境の作成
python -m venv venv

# 仮想環境の有効化
source venv/bin/activate  # Unix/macOS
# または
.\venv\Scripts\activate  # Windows

# 依存関係のインストール
pip install -r requirements.txt

## データの前処理

前処理済みデータの生成手順：

1. データの配置
   - HAM10000の画像ファイルを `data/raw/` に配置
   - メタデータCSVファイルを `data/raw/` に配置

2. 前処理の実行
   ```bash
   # Jupyter Notebookを起動
   jupyter notebook

   # notebooks/01_data_analysis.ipynb を実行