import numpy as np
import pandas as pd

def generate_operating_points(u_range=(0.8, 1.2), i_range=(0.0, 1.0), step=0.1):
    """
    TODO: 当前为假数据，仅生成简单的网格点
    """
    u_vals = np.arange(u_range[0], u_range[1] + step, step)
    i_vals = np.arange(i_range[0], i_range[1] + step, step)
    
    points = []
    for u in u_vals:
        for i in i_vals:
            points.append({'U': round(u, 2), 'I': round(i, 2)})
    
    # 限制数量避免测试太慢，取前20个
    return pd.DataFrame(points[:20])