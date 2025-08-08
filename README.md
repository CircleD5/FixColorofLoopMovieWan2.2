# FixColorofLoopMovieWan2.2
[terracottahaniwa](https://huggingface.co/terracottahaniwa)氏のアイデアをもとに作成したComfyUIカスタムノードです。

Wan2.2のStartフレームとEndフレームを指定してループ動画を作った時、始点付近と終点付近の画像の色に差異が生じてしまう問題があります。

このノードは画像の始点付近と終点付近にフィルターをかけることで、その違和感を軽減します。

サンプルのワークフローは/sample/FixLoopMovieExample.jsonにあります。
