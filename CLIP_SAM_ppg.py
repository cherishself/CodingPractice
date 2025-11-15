import numpy as np
from ..utils.morphology import make_kernel, dilate

def select_topk_points(score_map, k, min_spacing=400):
    # 贪心选取 Top-k，满足最小间距
    ys, xs = np.where(score_map > 0)
    vals = score_map[ys, xs]
    order = np.argsort(-vals)
    points = []
    for idx in order:
        y, x, v = ys[idx], xs[idx], vals[idx]
        ok = True
        for (px, py) in points:
            if (px - x)**2 + (py - y)**2 < min_spacing**2:
                ok = False; break
        if ok:
            points.append((x, y))
        if len(points) >= k:
            break
    return points

def co_feature_point_prompt_generation(Sa, Mapa, sam_feats, cfg):
    # Positive points
    Ra = (Sa > 0).astype(np.uint8) & (Mapa > 0).astype(np.uint8)
    pos_points = select_topk_points(Sa * Ra, cfg['k_top_pos'], cfg['min_point_spacing'])

    kernel = make_kernel(cfg['dilation']['shape'], tuple(cfg['dilation']['size']))
    Sa_bin = (Sa > 0).astype(np.uint8)
    ring = (dilate(Sa_bin, kernel) - Sa_bin).clip(0,1)
    # 局部特征
    Fa = sam_feats * Sa_bin[..., None]
    Fn = sam_feats * ring[..., None]
    # 余弦相似性近似
    eps = 1e-8
    Fa_mean = Fa.mean(axis=-1)
    Fn_mean = Fn.mean(axis=-1)
    num = (Fa_mean * Fn_mean)
    den = np.sqrt((Fa_mean**2).sum() + eps) * np.sqrt((Fn_mean**2).sum() + eps)
    sim_map = num / (den + eps)
    # 选相似度最低的 k
    neg_points = select_topk_points((1.0 - sim_map) * ring, cfg['k_low_neg'], cfg['min_point_spacing'])
    return pos_points, neg_points
