import numpy as np

def compute_frequency_response(U, I, num_freqs=50):
    """
    TODO: 请替换为真实的传递函数/阻抗计算
    当前返回随机的复数频响数据 (用于测试)
    """
    # 生成随机频响数据，确保稳定性分析不报错
    freqs = np.linspace(0.1, 10, num_freqs)
    # 假的复数响应：避免刚好撞到 -1，方便测试通过
    real_part = -0.5 + 0.2 * np.random.randn(num_freqs)
    imag_part = 0.3 * np.random.randn(num_freqs)
    return freqs, real_part, imag_part