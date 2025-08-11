# はじめてのPythonデータ処理（再スタート用ミニ雛形）

> ここからは **VS Code を入れた直後** からやる前提。  
> すべて **「どこに入力するか」** を明記しています。

---

## 0) 構成（このフォルダの中身）
```
src/etl.py            # 実行するスクリプト
data/raw/sample.csv   # 入力データ（小さなサンプル）
data/processed/       # 出力が入る
artifacts/            # 図が入る
tests/test_etl.py     # 最小テスト（後で使う）
requirements.txt      # 使う道具の一覧
.vscode/settings.json # VS Code が .venv を自動選択する設定
```

---

## 1) 仮想環境フォルダを作る・入る（VS Code のターミナル）
**入力先**：VS Code 下のターミナル（PowerShell）

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

行頭に `(.venv)` が付けばOK。

---

## 2) 必要な道具を入れる（VS Code のターミナル）
```powershell
pip install -r requirements.txt
```

---

## 3) 動かす（VS Code のターミナル）
```powershell
python -m src.etl --in data/raw/sample.csv --out data/processed/summary.csv --fig artifacts/eda_fig.png --verbose --show
```
- 進捗が出ます（[1/4] …）。
- 画像ウィンドウが開きます（閉じれば終了）。
- `data/processed/summary.csv` と `artifacts/eda_fig.png` が作られます。

---

## 4) うまくいかない時（よくある3つ）
- `unrecognized arguments` → コード保存漏れ／引数の打ち間違い
- `No module named 'src'` → ターミナルの位置が違う（このフォルダを開いてやり直す）
- スクリプトが無言 → `--verbose` を付ける or `Get-Content data/processed/summary.csv -TotalCount 5` で中身確認

---

## 5) （任意）Git で履歴を残す
**入力先**：VS Code のターミナル

```powershell
git init
git add .
git commit -m "first run"
# GitHub を使うなら：新規リポジトリを作成→URLをコピーして
git remote add origin <GitHubのURL>
git branch -M main
git push -u origin main
```

> まずは動かすことが最優先。Git/Docker は **後から** でもOK。

---

## 6) （任意）テストを動かす（VS Code のターミナル）
```powershell
python -m pytest -q
```

---

## 用語：どこに入力？
- **VS Code のターミナル**：コマンドを打つ場所（PowerShell）。
- **VS Code のエディタ**：`src/etl.py` など **コードを書く場所**。

困ったら、**打ったコマンドと赤字の最後の数行**を貼ってください。最短で直します。
