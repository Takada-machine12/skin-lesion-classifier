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
### 仮想環境の作成
```bash
python -m venv venv
```
### 仮想環境の有効化
```bash
source venv/bin/activate  # Unix/macOS
# または
.\venv\Scripts\activate  # Windows
```
### 依存関係のインストール
```bash
pip install -r requirements.txt
```
### データの前処理

前処理済みデータの生成手順：

1. データの配置
   - HAM10000の画像ファイルを `data/raw/` に配置
   - メタデータCSVファイルを `data/raw/` に配置

2. 前処理の実行
```bash
   # Jupyter Notebookを起動
   jupyter notebook

   # notebooks/01_data_analysis.ipynb を実行
```
## アプリケーションの使用方法
### アプリケーションの起動
```bash
steamlit run src/app/main.py
```
## プロジェクト構造
```bash
skin-lesion-classifier/
├── data/          # データ関連ファイル
│   ├── raw/      # 元データ
│   ├── processed/ # 処理済みデータ
│   └── test/     # テスト用データ
├── models/        # 学習済みモデル
├── notebooks/     # 分析・学習用ノートブック
├── src/          # ソースコード
└── tests/        # テストコード
```
## テストの実行方法
### 全てのテストを実行
```bash
python -m unittest discover tests
```
### 個別のテストを実行
```bash
python -m unittest tests/test_preprocessing.py
python -m unittest tests/test_model.py
python -m unittest tests/test_app.py
```
## モデルの性能
現在のモデルは以下の性能を達成しています。
・訓練データでの精度：約80%
・検証データでの精度：約70%
・テストデータでの性能
   ・良性症例の判定精度：75%以上
   ・悪性症例の判定精度：53%以上

※注意：このモデルは研究・開発目的で作成されており、実際の医療診断には使用できません。