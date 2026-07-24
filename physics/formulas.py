import numpy as np

def compute_frequency_response(U, I, num_freqs=50):
    """
    TODO: 当前返回随机的复数频响数据 (用于测试)，原项目使用较为复杂的导纳公式
    """
    # 生成随机频响数据，确保稳定性分析不报错
    freqs = np.linspace(0.1, 10, num_freqs)
    real_part = -0.5 + 0.2 * np.random.randn(num_freqs)
    imag_part = 0.3 * np.random.randn(num_freqs)
    return freqs, real_part, imag_part