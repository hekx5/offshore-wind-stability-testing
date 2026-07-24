import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_stable_case():
    """稳定的测试用例：频响曲线不包围 -1 点"""
    # 构造一个稳定的频响（实部大于 -1 且不环绕）
    real = np.array([-0.2, 0.1, 0.5, 0.3, -0.1])
    imag = np.array([0.1, -0.2, 0.3, -0.1, 0.0])
    return real, imag

@pytest.fixture
def sample_unstable_case():
    """不稳定的测试用例：频响曲线包围 -1 点"""
    # 构造一个不稳定的频响（跨越 -1 左侧）
    real = np.array([-2.0, -1.5, 1.0, 1.5, -1.8])
    imag = np.array([0.5, -0.5, 0.2, -0.2, 0.6])
    return real, imag

@pytest.fixture
def sample_training_data():
    """用于模型训练的假数据集"""
    np.random.seed(42)
    data = {
        'U': np.random.uniform(0.8, 1.2, 50),
        'I': np.random.uniform(0.0, 1.0, 50),
        'label': np.random.randint(0, 2, 50)  # 0稳定, 1不稳定
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_test_data():
    """用于模型评估的假测试集"""
    np.random.seed(123)
    data = {
        'U': np.random.uniform(0.8, 1.2, 20),
        'I': np.random.uniform(0.0, 1.0, 20),
        'label': np.random.randint(0, 2, 20)
    }
    return pd.DataFrame(data)