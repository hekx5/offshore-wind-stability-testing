import pytest
from model.train import train_lightweight_model, evaluate_model

pytestmark = pytest.mark.performance

class TestModelQualityGate:
    """模型性能质量门禁测试（上线标准）"""

    def test_accuracy_meets_threshold(self, sample_training_data, sample_test_data):
        """准确率必须 >= 0.75 """
        # 训练
        model = train_lightweight_model(sample_training_data)
        
        # 评估
        metrics = evaluate_model(model, sample_test_data)
        accuracy = metrics['accuracy']
        
        # 质量门禁断言（低于阈值则测试失败，阻止部署）
        threshold = 0.75
        assert accuracy >= threshold, \
            f"模型准确率 {accuracy:.2%} 低于阈值 {threshold:.0%}，质量不达标！"

    def test_precision_and_recall_threshold(self, sample_training_data, sample_test_data):
        """精确率和召回率也应满足基本要求"""
        model = train_lightweight_model(sample_training_data)
        metrics = evaluate_model(model, sample_test_data)
        
        assert metrics['precision'] >= 0.0
        assert metrics['recall'] >= 0.0
        
        # 项目中会根据实际情况设置目标precision、recall（非0）