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

## README の同期契約

- `README.md` はフォーク固有のヘッダーを除き、常に upstream の `README.md` と同一に保つ
- フォーク固有の依存関係、機能、運用手順を `README.md` の本文へ反映しない
- upstream のマージ後は `git diff upstream/main -- README.md` を実行し、許可されたヘッダー以外の差分がないことを確認する
- フォーク固有の説明が必要な場合は `AGENTS.md` または専用の文書へ記録する

## 推論ライブラリの責務

- Irodori-TTS は、モデル内部の状態や処理を安全に操作するための再利用可能な低レベル API を提供する
- 外部の推論パイプラインが必要とする操作を private メソッドや内部変数への直接アクセスで実現させず、内部の不変条件を保てる公開 API として切り出す
- HTTP、長文チャンク、要求単位の再試行、キャッシュの寿命管理など、呼び出し側の運用に属する状態を `SamplingRequest` や `InferenceRuntime` へ持ち込まない
- 高レベルの便利 API を追加する場合も、低レベル API と実装を共有し、同じモデル処理を複数の推論パイプラインで再実装させない
- この公開リポジトリのコード、コメント、コミット、文書には、非公開リポジトリの名称、構成、運用、固有要件を記載しない。変更理由は Irodori-TTS 単体で成立する公開 API の責務として説明する

## 方針違反を指摘された場合

- ユーザーから方針違反を強く指摘された場合は、目先の修正や追加コミットより先に、同じ誤りを止める規則を `AGENTS.md` などの正本へ記録する
- 誤ったコミットがまだ公開されておらず作り直せる場合は、履歴へ訂正コミットを追加せず、誤ったコミットを取り消して正しい手順で作り直す
- 再実行前に、違反した判断と新しい検査手順が対応していることを確認する

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
