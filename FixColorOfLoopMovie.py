import torch
import numpy as np

class Int2String:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "i": ("INT", {"default": 0})
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("str", )
    FUNCTION = "i2s"
    CATEGORY = "Circle"
    def i2s(self, i):
        return str(i)


class FixColor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # [B,H,W,C], float32, 0..1
                "d1": ("INT", {"default": 10, "min": 1}),
                "d2": ("INT", {"default": 10, "min": 1})
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )
    FUNCTION = "fix_color_direct"
    CATEGORY = "Circle"

    def fix_color_direct(self, image, d1, d2):
        # image: [B,H,W,C], float32, 0..1
        result_image = image.clone()
        L = result_image.shape[0]
        eps = 1e-6

        # 安全チェック
        if L == 0 or d1 < 1 or d2 < 1 or (d1 + d2) > L:
            return (result_image, )

        # 各フレームの平均値と標準偏差を計算
        mean_vals = []
        std_vals  = []
        for i in range(L):
            f = result_image[i]  # [H,W,C]
            flat = f.reshape(-1, 3)
            mu = flat.mean(dim=0)
            sd = flat.std(dim=0, unbiased=False)
            mean_vals.append(mu)
            std_vals.append(sd)

        mean_vals = torch.stack(mean_vals, dim=0)  # (L,3)
        std_vals  = torch.stack(std_vals,  dim=0)  # (L,3)

        # 対象範囲インデックス
        seg_len = d1 + d2
        start_idx = torch.arange(0, seg_len, dtype=torch.long)
        end_idx   = torch.arange(L - seg_len, L, dtype=torch.long)

        # target_mean / target_std は両端区間から算出
        boundary_idx = torch.cat([start_idx, end_idx], dim=0)
        target_mean = mean_vals[boundary_idx].mean(dim=0)  # (3,)
        target_std  = std_vals[boundary_idx].mean(dim=0)   # (3,)

        # 線形フェード関数
        def weight_start(i):
            if i <= d1 - 1:
                return 1.0
            elif i <= d1 + d2 - 1:
                return max(0.0, 1.0 - (i - (d1 - 1)) / float(d2))
            else:
                return 0.0

        def weight_end(i):
            start = L - (d1 + d2)
            knee  = L - d1
            if i < start:
                return 0.0
            elif i <= knee:
                return (i - start) / float(d2)
            else:
                return 1.0

        # 補正処理
        for i in range(L):
            in_start = (0 <= i <= (seg_len - 1))
            in_end   = (L - seg_len <= i <= L - 1)
            if not (in_start or in_end):
                continue

            arr = result_image[i]
            H, W, C = arr.shape
            flat = arr.reshape(-1, 3)

            mu = mean_vals[i]
            sd = std_vals[i]

            if in_start:
                w = weight_start(i)
            else:
                w = weight_end(i)

            if w <= 0.0:
                continue

            # targetへ寄せる補正
            target_mean_frame = mu + w * (target_mean - mu)
            target_std_frame  = sd + w * (target_std - sd)

            arr_std = (flat - mu) / (sd + eps)
            arr_corr = arr_std * target_std_frame + target_mean_frame
            arr_corr = torch.clamp(arr_corr, 0.0, 1.0).reshape(H, W, C)

            result_image[i] = arr_corr

        return (result_image, )
