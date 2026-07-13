# MicMeas / OtoScope

ブラウザ完結の音測定・埋め込み観測ノート（PoC）。

- マイクで録音 → **端末内**で YAMNet（意味的）＋ DSP（物理的）埋め込み
- IndexedDB に蓄積・可視化・簡易定点観測
- モデルも同梱配信のため、**実行時の外部通信はゼロ**
- GitHub Pages（HTTPS）向け静的アプリ

**公開 URL（Pages 有効化後）:** https://smyrk2031.github.io/MicMeas/

## 使い方

1. ☰設定でシリーズ（測定場所・対象）を作る
2. 「音測定スタート」→ 数秒録音 → 類似度・品質警告・レコメンド
3. 「📊 レコード解析」でカラーマップ / 埋め込みマップ / 一覧
4. 詳細モーダルで再生・帯域マーク・異常メモ
5. 「JSONエクスポート」でバックアップ（音声は含まれません）

スマホではブラウザメニューの「ホーム画面に追加」で PWA として使えます。

## ローカルで試す

静的ファイルだけなので、Python の簡易サーバでも起動できます。

```bash
# リポジトリ直下で
python -m http.server 8000
# → http://localhost:8000
```

マイクは **localhost または HTTPS** でのみ動作します（ブラウザ仕様）。

## 構成

| パス | 役割 |
|---|---|
| `index.html` / `app.js` / `style.css` | UI・録音・埋め込み・可視化 |
| `vendor/tf.min.js` | TensorFlow.js（自己ホスト） |
| `models/yamnet/` | YAMNet モデル一式 |
| `sw.js` / `manifest.webmanifest` | PWA |
| `.nojekyll` | GitHub Pages で `_` 等をそのまま配信 |

## License

MIT
