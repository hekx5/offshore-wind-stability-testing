import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_lightweight_model(data: pd.DataFrame, epochs=1):
    """
    TODO: 替换为你的神经网络（如PMGM）或复杂训练逻辑
    当前使用极简 sklearn 随机森林，保证测试可跑
    """
    # 假设数据包含 'U', 'I', 'label' 三列
    if 'label' not in data.columns:
        # 如果没有标签，生成随机标签
        data['label'] = [0] * len(data)
    
    X = data[['U', 'I']].values
    y = data['label'].values
    
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    return model

def evaluate_model(model, test_data):
    """返回准确率、召回率等指标（用于性能测试）"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    X = test_data[['U', 'I']].values
    y_true = test_data['label'].values
    y_pred = model.predict(X)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec}