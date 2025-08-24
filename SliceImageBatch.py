# file: slice_image_batch.py
# ComfyUI custom node: SliceImageBatch
# ImageBatch([B,H,W,C]) から [start, end) を返す（end は含まない）
# 変更点:
#  - end_index < 0 のとき「末尾から -end_index 枚を削る」挙動に統一
#  - clone_tensor を廃止し、常に .clone() したテンソルを返す

import torch

class SliceImageBatch:
    """
    ComfyUI node to slice an ImageBatch by start (inclusive) and end (exclusive).
    end_index < 0 means "drop -end_index frames from the tail".
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # [B,H,W,C] float32 0..1
                "start_index": ("INT", {
                    "default": 0,
                    "min": -10_000,
                    "max": 10_000
                }),
                "end_index": ("INT", {
                    "default": -1,   # -1: 末尾から1枚削除（= B-1 まで）
                    "min": -10_000,
                    "max": 10_000
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "slice_batch"
    CATEGORY = "Circle/Batch"

    @staticmethod
    def _normalize_indices(b: int, start: int, end: int):
        """
        インデックスを Python スライス風に正規化しつつ、
        end < 0 は「末尾から -end 枚削る」扱いにする。
        """
        # start: 通常の負インデックス対応（後ろから数える）
        if start < 0:
            start = b + start

        # end: 通常の負インデックス（b+end）= 末尾から -end 枚を削る地点
        if end < 0:
            end = b + end  # 例: end=-1 -> b-1（= 後ろ1枚削る）

        # [0, b] にクランプ
        start = max(0, min(start, b))
        end = max(0, min(end, b))

        return start, end

    def slice_batch(self, image, start_index: int, end_index: int):
        if not torch.is_tensor(image):
            raise TypeError("`image` must be a torch.Tensor shaped [B,H,W,C].")

        if image.ndim != 4:
            raise ValueError(f"`image` must be 4D [B,H,W,C], got shape {tuple(image.shape)}")

        b = int(image.shape[0])
        if b == 0:
            raise ValueError("Empty ImageBatch (B==0). Nothing to slice.")

        start, end = self._normalize_indices(b, start_index, end_index)

        if end <= start:
            raise ValueError(
                f"Invalid slice range: start={start}, end={end}. "
                "After normalization, start must be < end (non-empty slice)."
            )

        out = image[start:end].clone()  # 常に clone
        return (out,)

