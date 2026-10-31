---
title: Koharu のインストール
description: Koharu をローカルでビルドしてインストールします。
---

# Koharu のインストール

このフォークはソースからローカルでビルドします。[開発環境のセットアップ](/ja-JP/development/setup/)に記載された依存関係をインストールしてから、次を実行します。

```powershell
bun install
cargo tauri build --bundles nsis
```

Windows インストーラーは `target/release/bundle/nsis` に出力されます。

## 初回起動

ネイティブランタイムの準備が完了すると、プロジェクトブラウザーが表示されます。初回はネイティブパッケージをダウンロードするため、通常より時間がかかることがあります。各モデルのファイルは、そのモデルを初めて使用するときに解決されます。

ダウンロードには GitHub のリリースアセットと、モデル重みの場合は Hugging Face への接続が必要です。進捗はアクティビティセンターに表示されます。パッケージをローカルキャッシュへ公開している途中で終了しないでください。

次は[最初のプロジェクト](/ja-JP/getting-started/first-project/)を作成します。ハードウェア選択とキャッシュについては[ランタイム、モデル、ハードウェア](/ja-JP/getting-started/runtime-models-and-hardware/)を参照してください。
