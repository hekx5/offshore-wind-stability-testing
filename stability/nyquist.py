import numpy as np

def compute_winding_number(real, imag):
    """
    TODO: 当前假逻辑：简单根据实部均值判断（项目中替换其他判断方法）
    """
    # 假逻辑：如果平均实部大于 -0.5 认为稳定
    return 0 if np.mean(real) > -0.5 else 1

def is_stable(real_part, imag_part):
    """
    判稳主函数
    输入：频响的实部和虚部 (np.array)
    输出：True(稳定) / False(不稳定)
    """
    if real_part is None or imag_part is None or len(real_part) == 0:
        raise ValueError("Invalid frequency response input")
    
    # 调用绕圈数计算
    winding = compute_winding_number(real_part, imag_part)
    # 奈奎斯特判据：绕 (-1,0j) 点圈数为0则稳定
    return winding == 0