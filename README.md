# FixColorofLoopMovieWan2.2
[terracottahaniwa](https://huggingface.co/terracottahaniwa)氏のアイデアをもとに作成したComfyUIカスタムノードです。

Wan2.2のStartフレームとEndフレームを指定してループ動画を作った時、始点付近と終点付近の画像の色に差異が生じてしまう問題があります。

このノードは画像の始点付近と終点付近にフィルターをかけることで、その違和感を軽減します。

サンプルのワークフローは/sample/FixLoopMovieExample.jsonにあります。

パラメータ
- image : ComfyUIのimage
- d1 : 始点（または終点）からどの範囲までフィルターを書けるか？（単位はフレーム数）
- d2 : フィルターの減衰範囲

例えば、[0,d1]までの範囲は強いフィルターをかけ、[d1,d1+d2]の範囲は徐々にフィルターの効果を弱めるというイメージ。
