import pytest
import numpy as np
from stability.nyquist import is_stable, compute_winding_number

pytestmark = pytest.mark.unit

class TestNyquistStability:
    """奈奎斯特判稳单元测试套件"""

    # 1. 参数化测试：覆盖多种正常情况
    @pytest.mark.parametrize("real, imag, expected", [
        # 稳定案例：实部整体偏正
        ([-0.1, 0.2, 0.3], [0.1, -0.1, 0.2], True),
        # 不稳定案例：实部大幅偏负
        ([-2.0, -1.8, 0.5], [0.5, -0.5, 0.1], False),
        # 临界稳定：正好经过 (-1, 0) 点附近（视为稳定）
        ([-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], True),
    ])
    def test_stability_logic(self, real, imag, expected):
        """测试判稳核心逻辑是否正确"""
        result = is_stable(np.array(real), np.array(imag))
        assert result == expected, f"期望 {expected}, 实际得到 {result}"

    # 2. 边界值分析：空数组
    def test_empty_input_raises_exception(self):
        """输入空数组应抛出 ValueError"""
        with pytest.raises(ValueError, match="Invalid frequency response"):
            is_stable(np.array([]), np.array([]))

    # 3. 异常值测试：None 输入
    def test_none_input_raises_exception(self):
        with pytest.raises(ValueError):
            is_stable(None, None)

    # 4. 类型检查：输入必须是数值
    def test_non_numeric_input_raises(self):
        with pytest.raises(Exception):  # 可能会因为计算报错
            is_stable(["a", "b"], [1, 2])

    # 5. 使用 Fixture 测试稳定样本
    def test_stable_sample_fixture(self, sample_stable_case):
        real, imag = sample_stable_case
        assert is_stable(real, imag) is True

    # 6. 使用 Fixture 测试不稳定样本
    def test_unstable_sample_fixture(self, sample_unstable_case):
        real, imag = sample_unstable_case
        assert is_stable(real, imag) is False

    # 7. 绕圈数计算（子函数）的独立测试
    def test_winding_number_zero(self):
        real = np.array([0.5, 0.6, 0.7])
        imag = np.array([0.1, -0.1, 0.2])
        # 当前假逻辑，只要均值大于-0.5就返回0
        assert compute_winding_number(real, imag) == 0

    def test_winding_number_one(self):
        real = np.array([-2.0, -1.0])
        imag = np.array([0.5, -0.5])
        # 假逻辑返回1
        assert compute_winding_number(real, imag) == 1