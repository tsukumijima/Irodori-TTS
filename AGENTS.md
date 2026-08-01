# AGENTS.md

このリポジトリは [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) のフォークです。  
AI コーディングエージェント向けに、ブランチ運用とコントリビューションのルールを以下に定めます。

## ブランチ戦略

### `main` ブランチ — upstream 追従用

- upstream 本家 (`Aratako/Irodori-TTS`) へプルリクエストを送る際の基底ブランチ
- upstream の更新があれば日常的に pull (fast-forward or merge) して追従する
- 基本的にはプレーンな upstream のコードがそのまま含まれる状態を維持する
  - PR が本家に取り込まれれば、マージコミットを除きコード差分はほぼゼロになる想定
  - このリポジトリ固有の CI 設定 (GitHub Actions) など、upstream に存在しない管理系ファイルが含まれることはあり得る

### `master` ブランチ — 独自フォークのメインブランチ

- このリポジトリのデフォルトブランチであり、独自に保守・発展させていくブランチ
- upstream に取り込んでもらう予定のない変更は、基本的にこちらに直接コミットしていく
  - 汎用性が低く自分用途に特化した機能追加
  - 一定の副作用やスタイルの好みがあり upstream へ提案するほどではない改善
  - その他、独自に積極的に改良していきたい変更全般
- pyproject.toml は自環境向けに大幅にカスタムされており、upstream の構成とは大きく異なる
  - 依存関係の管理は uv に統一しており、requirements.txt は存在しない
  - master ブランチに変更を加えた後は、必ず `uv run task lint` / `uv run task format` / `uv run task test` を実行し、コード品質をチェックすること

## upstream への貢献フロー

### 最初から upstream 貢献を意図する場合

1. `main` ブランチの最新から `feature/(適切な名前)` ブランチを切る
2. feature ブランチ上で PR 用の変更を重ねる
3. PR がマージされるかに関わらず、一旦 `master` ブランチにもマージして実利用する
4. upstream の PR がマージされたら `main` を pull して追従する

### master で実装した変更を後から upstream に投げる場合

`master` ブランチで実装・検証した変更を後から upstream へ提案したくなるケースもあり得る。  
この場合は以下のいずれかの方法で PR 用ブランチを作成する。

- **cherry-pick が綺麗に適用できる場合:** `main` の最新コミットをベースに、当該変更のコミットを cherry-pick したブランチを生やす
- **cherry-pick では困難な場合 (変更が複数コミットに散らばっている等):** `master` ブランチでの変更のうち、当該機能・修正に関する差分だけを抽出し、`main` から `feature/(適切な名前)` ブランチを切って手動で適用する

## コミットメッセージ規約

コミットメッセージは以下のフォーマットに従う。

- 言語: **英語**
- Prefix: コミット内容に応じて以下のいずれかを付与する
  - `Add:` — 新機能の追加
  - `Update:` — 既存機能の改善・更新
  - `Fix:` — バグ修正
  - `Refactor:` — リファクタリング (機能変更なし)
  - `Remove:` — コード・ファイルの削除

### 例

```
Add: Speaker inversion training and inference pipeline
Update: Improve audio quality of watermark embedding
Fix: Incorrect tensor shape in duration predictor
Refactor: Extract shared utility functions from model.py
Remove: Deprecated legacy vocoder support
```
