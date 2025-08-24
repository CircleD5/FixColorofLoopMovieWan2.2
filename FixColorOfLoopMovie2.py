import torch

class FixColorHighPrecision:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # [B,H,W,C], float32, 0..1
                "d1": ("INT", {"default": 10, "min": 1}),
                "d2": ("INT", {"default": 10, "min": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )
    FUNCTION = "fix_color_cov"
    CATEGORY = "Circle"

    # ---- 全画素から平均・共分散（厳密：二次モーメント） ----
    @staticmethod
    def _mean_cov_from_flat_all(flat_f32: torch.Tensor, eps: float = 1e-6):
        """
        flat_f32: [N,3] float32
        μ = E[x],  C = E[xx^T] - μ μ^T を 64bit 積算で安定に計算。
        """
        x = flat_f32.to(torch.float64)
        N = x.shape[0]
        S1 = x.sum(dim=0)          # (3,)
        S2 = x.t().mm(x)           # (3,3)
        mu = S1 / float(N)         # (3,)
        C  = (S2 / float(N)) - torch.outer(mu, mu)  # (3,3)
        # 数値安定の下駄
        C = C + eps * torch.eye(3, dtype=C.dtype, device=C.device)
        return mu.to(flat_f32.dtype), C.to(flat_f32.dtype)

    @staticmethod
    def _sqrtm_psd(C: torch.Tensor, eps: float = 1e-6, inverse: bool = False):
        """
        C: [3,3] 対称(半)正定値の（逆）平方根。固有分解＋下限クリップ。
        """
        Cd = C.to(torch.float64)
        vals, vecs = torch.linalg.eigh(Cd)     # 昇順
        vals = torch.clamp(vals, min=eps)
        if inverse:
            S = torch.diag(torch.rsqrt(vals))
        else:
            S = torch.diag(torch.sqrt(vals))
        M = vecs @ S @ vecs.T
        return M.to(C.dtype)

    def fix_color_cov(self, image, d1, d2):
        """
        共分散まで一致させる WCT（全画素使用）＋ ランプ適用。
        - ターゲット(μ_t, C_t): 先頭(d1+d2) と末尾(d1+d2) の “全画素” から厳密算出
        - 各フレーム i: 全画素から (μ_i, C_i) を厳密算出 → y = μ_t + C_t^{1/2} C_i^{-1/2} (x - μ_i)
        - ランプ w(i) で (1-w)·x + w·y を適用（先頭側はフェードアウト、末尾側はフェードイン）
        """
        result_image = image.clone()
        L, H, W, C = result_image.shape
        if L == 0 or d1 < 1 or d2 < 1 or (d1 + d2) > L or C != 3:
            return (result_image, )

        device = result_image.device
        dtype  = result_image.dtype
        eps = 1e-6

        seg_len = d1 + d2
        start_idx = torch.arange(0, seg_len, dtype=torch.long, device=device)
        end_idx   = torch.arange(L - seg_len, L, dtype=torch.long, device=device)
        boundary_idx = torch.cat([start_idx, end_idx], dim=0)

        # ---- ターゲット統計（全画素厳密）：二次モーメント逐次加算 ----
        total_N = 0
        S1 = torch.zeros(3, dtype=torch.float64, device=device)
        S2 = torch.zeros(3, 3, dtype=torch.float64, device=device)
        for i in boundary_idx.tolist():
            flat64 = result_image[i].reshape(-1, 3).to(torch.float64)  # [N,3]
            total_N += flat64.shape[0]
            S1 += flat64.sum(dim=0)
            S2 += flat64.t().mm(flat64)

        mu_t = (S1 / float(total_N)).to(dtype)                       # (3,)
        C_t  = (S2 / float(total_N)).to(dtype) - torch.outer(mu_t.to(torch.float64),
                                                             mu_t.to(torch.float64)).to(dtype)
        C_t  = C_t + eps * torch.eye(3, dtype=dtype, device=device)  # 安定化
        Ct_sqrt = self._sqrtm_psd(C_t, eps=eps, inverse=False)       # (3,3)

        # ---- ランプ関数 ----
        def weight_start(i):
            # 先頭側：d1で1.0、次のd2で線形に0へ
            if i <= d1 - 1:
                return 1.0
            elif i <= d1 + d2 - 1:
                return max(0.0, 1.0 - (i - (d1 - 1)) / float(d2))
            else:
                return 0.0

        def weight_end(i):
            # 末尾側：d2で0→1、最後のd1で1.0
            start = L - (d1 + d2)
            knee  = L - d1
            if i < start:
                return 0.0
            elif i <= knee:
                return (i - start) / float(d2)
            else:
                return 1.0

        # ---- 各フレームに適用（全画素で μ_i, C_i 計算） ----
        for i in range(L):
            in_start = (0 <= i <= (seg_len - 1))
            in_end   = (L - seg_len <= i <= L - 1)
            if not (in_start or in_end):
                continue

            w = weight_start(i) if in_start else weight_end(i)
            if w <= 0.0:
                continue

            arr  = result_image[i]                 # [H,W,3]
            flat = arr.reshape(-1, 3)              # [N,3] float32

            # フレーム i の厳密統計
            mu_i, C_i = self._mean_cov_from_flat_all(flat, eps=eps)  # (3,), (3,3)
            Ci_invsqrt = self._sqrtm_psd(C_i, eps=eps, inverse=True) # (3,3)

            # y_full = μ_t + C_t^{1/2} C_i^{-1/2} (x - μ_i)
            y_full = (flat - mu_i) @ Ci_invsqrt.T
            y_full = y_full @ Ct_sqrt.T + mu_t

            # ランプ適用（画素空間で線形ブレンド）
            arr_corr = (1.0 - w) * flat + w * y_full
            arr_corr = torch.clamp(arr_corr, 0.0, 1.0).reshape(H, W, 3)

            result_image[i] = arr_corr

        return (result_image, )
