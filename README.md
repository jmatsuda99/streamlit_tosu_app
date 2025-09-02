# 鳥栖 需要分析ツール（GitHub + Streamlit）

## 使い方
1. このフォルダをGitHubにpush（メインファイルは `app.py`）
2. Streamlit CloudでNew app → リポジトリ/ブランチ/メインファイルを指定
3. 左サイドバーでExcelをアップロード → DB保存 → 解析対象を選択
4. 6〜8の予測はチェックで実行（LightGBMが無い環境は自動でGBDTにフォールバック）

## ヒント（白画面対策）
- デプロイ設定の **Main file path** が `app.py` になっているか確認
- **Logs** を開き、ModuleNotFoundError（依存漏れ）、SyntaxError を確認
- `requirements.txt` をリポジトリ直下に置く
- 依存が重い場合は `lightgbm` を外してデプロイ → 後で追加