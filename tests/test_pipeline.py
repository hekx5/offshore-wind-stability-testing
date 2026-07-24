import pytest
from physics.sampler import generate_operating_points
from physics.formulas import compute_frequency_response
from stability.nyquist import is_stable
from model.train import train_lightweight_model

pytestmark = pytest.mark.integration

class TestDataPipeline:
    """数据流端到端集成测试（冒烟测试）"""

    def test_full_pipeline_no_crash(self):
        """全流程冒烟测试：生成 -> 判稳 -> 训练，确保不报错"""
        # 1. 采样（使用小步长快速生成）
        raw_df = generate_operating_points(
            u_range=(0.9, 1.1),
            i_range=(0.2, 0.4),
            step=0.2  # 大步长，只生成少量数据
        )
        assert len(raw_df) > 0

        # 2. 判稳（模拟给每条数据打标签）
        labels = []
        for _, row in raw_df.iterrows():
            # 调用物理公式（虽然是假的）
            freqs, real, imag = compute_frequency_response(row['U'], row['I'])
            # 判稳
            stable = is_stable(real, imag)
            labels.append(1 if stable else 0)  # 模型需要0/1格式
        
        raw_df['label'] = labels
        assert 'label' in raw_df.columns

        # 3. 极简训练
        model = train_lightweight_model(raw_df, epochs=1)
        assert model is not None

    def test_data_generation_has_required_columns(self):
        """检查生成的数据是否包含必要的 U 和 I 列"""
        df = generate_operating_points(step=0.5)
        assert 'U' in df.columns
        assert 'I' in df.columns

    def test_nyquist_called_with_correct_types(self):
        """检查判稳函数是否能接受列表/数组输入"""
        freqs, real, imag = compute_frequency_response(1.0, 0.5)
        # 判稳函数应该能处理 numpy array
        result = is_stable(np.array(real), np.array(imag))
        # 只是验证不抛异常
        assert isinstance(result, bool)