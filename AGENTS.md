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
- フォーク固有のパラメーターや使い方は `docs/parameters.md` へ記録し、設計・運用契約は `AGENTS.md` へ記録する
- `docs/parameters.md` の先頭には、現行 `master` のフォーク固有パラメーターを含む文書であることを明記する
- upstream のマージ後は `docs/parameters.md` も確認し、現行の CLI・設定フィールド・フォーク固有機能が全て説明されている状態を保つ

## 第三者コードの移植契約

- 第三者リポジトリからコードを移植する場合は、出典 URL・対象リビジョン・著作権表示・許諾文を移植ファイルの先頭へ記載する
- 移植元のコード、コメント、Docstring は原文と英語表記を可能な限り保ち、Irodori-TTS の実行契約に必要な最小限の変更だけを加える
- リポジトリの通常のコメント文体と Docstring 規約は新規実装に適用し、原文保持が必要な第三者コードには機械的に適用しない
- 依存削減や入力形状の限定などで原文を変更した箇所は、何を保った移植かが分かる英語コメントを近傍に残す

## 推論ライブラリの責務

- Irodori-TTS は、モデル内部の状態や処理を安全に操作するための再利用可能な低レベル API を提供する
- 外部の推論パイプラインが必要とする操作を private メソッドや内部変数への直接アクセスで実現させず、内部の不変条件を保てる公開 API として切り出す
- HTTP、長文チャンク、要求単位の再試行、キャッシュの寿命管理など、呼び出し側の運用に属する状態を `SamplingRequest` や `InferenceRuntime` へ持ち込まない
- 高レベルの便利 API を追加する場合も、低レベル API と実装を共有し、同じモデル処理を複数の推論パイプラインで再実装させない
- コード、コメント、コミット、文書には、Irodori-TTS 単体で説明できない外部システムの名称、構成、運用、固有要件を記載しない

## 長時間学習の再開契約

- Speaker Inversion の学習レシピは対象 checkpoint 系列ごとに分け、既存レシピのモデル構成を別系列へ合わせて上書きしない
- Speaker Inversion の定期保存では、推論用の軽量な `.speaker.safetensors` と、optimizer・scheduler・step・乱数・dataloader の状態を含む学習再開用 sidecar を同時に保存する
- `--resume` は学習再開用 sidecar から frozen base model と話者埋め込みを復元し、中断前と同じデータ順・乱数状態で残りの step を継続できる状態を維持する
- warm-start 用の `--speaker-inversion-init-embedding` は新しい最適化として扱い、完全再開の代替として案内しない

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

- upstream を取り込むマージコミットは Git が生成したメッセージをそのまま使い、通常コミット向けの Prefix や説明本文へ書き換えない
- rebase でマージを含む履歴を書き換えた場合は、完了後に各マージコミットの親とメッセージを旧履歴と照合する
- 通常コミットは単独で説明できる機能・修正単位へ分け、複数の独立した変更を1つの件名へ列挙しない

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
