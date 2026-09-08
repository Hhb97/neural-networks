import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# ==================== 1. 准备数据 ====================
print("=" * 60)
print("1. 准备数据")
print("=" * 60)

# 生成房价数据（面积、房龄、房间数 → 房价）
np.random.seed(42)
n_samples = 200

# 特征：面积 (40-200), 房龄 (1-20), 房间数 (2-6)
X = np.zeros((n_samples, 3))
X[:, 0] = np.random.uniform(40, 200, n_samples)      # 面积
X[:, 1] = np.random.uniform(1, 20, n_samples)         # 房龄
X[:, 2] = np.random.randint(2, 7, n_samples)          # 房间数

# 真实公式：房价 = 2.5*面积 - 3*房龄 + 20*房间数 + 噪声
y = 2.5 * X[:, 0] - 3.0 * X[:, 1] + 20.0 * X[:, 2] + np.random.randn(n_samples) * 15

# 划分数据集：训练 60%，验证 20%，测试 20%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"训练集: {X_train.shape[0]} 个样本")
print(f"验证集: {X_val.shape[0]} 个样本")
print(f"测试集: {X_test.shape[0]} 个样本")

# ==================== 2. 数据标准化 ====================
print("\n" + "=" * 60)
print("2. 数据标准化")
print("=" * 60)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# 转换为 PyTorch Tensor
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train_scaled).reshape(-1, 1)
X_val_t = torch.FloatTensor(X_val_scaled)
y_val_t = torch.FloatTensor(y_val_scaled).reshape(-1, 1)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test_scaled).reshape(-1, 1)

# 创建 DataLoader（批量训练）
batch_size = 16
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ==================== 3. 定义神经网络 ====================
print("\n" + "=" * 60)
print("3. 定义神经网络")
print("=" * 60)

class TwoLayerNet(nn.Module):
    """两层神经网络：输入 → 隐藏层(ReLU) → 输出层"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        # 隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 输出层
        self.fc2 = nn.Linear(hidden_size, output_size)
        # 初始化权重（He 初始化）
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        # 隐藏层：线性变换 + ReLU
        x = torch.relu(self.fc1(x))
        # 输出层：线性变换（回归任务不加激活）
        x = self.fc2(x)
        return x

# 实例化模型
input_size = 3
hidden_size = 10  # 隐藏层神经元数
output_size = 1

model = TwoLayerNet(input_size, hidden_size, output_size)
print(model)

# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params:,}")

# ==================== 4. 训练配置 ====================
print("\n" + "=" * 60)
print("4. 训练配置")
print("=" * 60)

# 超参数
learning_rate = 0.01
weight_decay = 0.001  # L2 正则化系数（越大惩罚越强）
epochs = 2000
patience = 100  # 早停耐心值

# 损失函数：MSE
criterion = nn.MSELoss()

# 优化器：Adam（带 L2 正则化 weight_decay）
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

print(f"学习率: {learning_rate}")
print(f"L2 正则化系数: {weight_decay}")
print(f"最大迭代次数: {epochs}")
print(f"早停耐心值: {patience}")

# ==================== 5. 训练循环 ====================
print("\n" + "=" * 60)
print("5. 开始训练（带早停）")
print("=" * 60)

train_losses = []
val_losses = []
best_val_loss = float('inf')
wait = 0

for epoch in range(epochs):
    # ----- 训练阶段 -----
    model.train()
    epoch_train_loss = 0
    for batch_X, batch_y in train_loader:
        # 前向传播
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        
        # 反向传播 + 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item() * batch_X.size(0)
    
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # ----- 验证阶段 -----
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_t)
        val_loss = criterion(y_val_pred, y_val_t).item()
        val_losses.append(val_loss)
    
    # ----- 早停检查 -----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        # 保存最佳模型
        best_model_state = model.state_dict()
    else:
        wait += 1
        if wait >= patience:
            print(f"\n⚠️ 早停触发！在第 {epoch+1} 轮停止训练")
            break
    
    # 打印进度
    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:4d}/{epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {val_loss:.6f}")

# 恢复最佳模型
model.load_state_dict(best_model_state)
print(f"\n✅ 训练完成！最佳验证损失: {best_val_loss:.6f}")

# ==================== 6. 可视化训练曲线 ====================
print("\n" + "=" * 60)
print("6. 训练曲线")
print("=" * 60)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.7)
plt.plot(val_losses, label='Val Loss', alpha=0.7)
plt.axvline(x=len(train_losses)-1, color='red', linestyle='--', label='Early Stop')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('训练 & 验证损失曲线')
plt.legend()
plt.grid(True, alpha=0.3)

# ==================== 7. 测试集评估 ====================
print("\n" + "=" * 60)
print("7. 测试集评估")
print("=" * 60)

model.eval()
with torch.no_grad():
    # 标准化空间预测
    y_train_pred_scaled = model(X_train_t)
    y_val_pred_scaled = model(X_val_t)
    y_test_pred_scaled = model(X_test_t)
    
    # 反标准化到原始尺度
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.numpy())
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.numpy())
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.numpy())

# 计算误差
def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return mse, mae, r2

train_mse, train_mae, train_r2 = calculate_metrics(y_train, y_train_pred.flatten())
val_mse, val_mae, val_r2 = calculate_metrics(y_val, y_val_pred.flatten())
test_mse, test_mae, test_r2 = calculate_metrics(y_test, y_test_pred.flatten())

print("数据集评估指标:")
print("-" * 50)
print(f"{'数据集':<10} {'MSE':<12} {'MAE':<12} {'R²':<10}")
print("-" * 50)
print(f"{'训练集':<10} {train_mse:<12.2f} {train_mae:<12.2f} {train_r2:<10.4f}")
print(f"{'验证集':<10} {val_mse:<12.2f} {val_mae:<12.2f} {val_r2:<10.4f}")
print(f"{'测试集':<10} {test_mse:<12.2f} {test_mae:<12.2f} {test_r2:<10.4f}")
print("-" * 50)

# 过拟合分析
ratio = test_mse / train_mse if train_mse > 0 else 0
print(f"\n过拟合分析:")
print(f"  测试/训练 MSE 比值: {ratio:.2f}")
if ratio > 1.5:
    print("  ⚠️ 结论: 存在明显过拟合！考虑增大 weight_decay 或减少隐藏层神经元")
elif ratio > 1.2:
    print("  ⚡ 结论: 轻微过拟合")
else:
    print("  ✅ 结论: 泛化良好！")

# ==================== 8. 预测 vs 真实值散点图 ====================
plt.subplot(1, 2, 2)
all_y = np.concatenate([y_train, y_val, y_test])
all_pred = np.concatenate([y_train_pred.flatten(), y_val_pred.flatten(), y_test_pred.flatten()])
plt.scatter(all_y, all_pred, alpha=0.5, s=20)
plt.plot([all_y.min(), all_y.max()], [all_y.min(), all_y.max()], 'r--', linewidth=2)
plt.xlabel('真实房价')
plt.ylabel('预测房价')
plt.title(f'预测 vs 真实 (测试集 R²={test_r2:.3f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results.png', dpi=150)
print("\n📊 训练曲线和预测图已保存为 'training_results.png'")
plt.show()

# ==================== 9. 打印模型权重和公式 ====================
print("\n" + "=" * 60)
print("8. 模型权重和公式")
print("=" * 60)

# 提取权重
fc1_weight = model.fc1.weight.detach().numpy()
fc1_bias = model.fc1.bias.detach().numpy()
fc2_weight = model.fc2.weight.detach().numpy()[0]
fc2_bias = model.fc2.bias.detach().numpy()[0]

print("\n【W1】输入→隐藏层 (3 × 10):")
for i in range(fc1_weight.shape[0]):
    print(f"  h{i+1} = ReLU(", end="")
    for j in range(fc1_weight.shape[1]):
        print(f"{fc1_weight[i][j]:.4f}·x{j+1}", end="")
        if j < fc1_weight.shape[1] - 1:
            print(" + ", end="")
    print(f" + {fc1_bias[i]:.4f})")

print(f"\n【W2】隐藏层→输出 (10 × 1):")
print("  ŷ = ", end="")
for i in range(len(fc2_weight)):
    print(f"{fc2_weight[i]:.4f}·h{i+1}", end="")
    if i < len(fc2_weight) - 1:
        print(" + ", end="")
print(f" + {fc2_bias:.4f}")

print(f"\n【完整公式】")
print("ŷ = ReLU(X × W1 + b1) × W2 + b2")
print("注: 所有特征已标准化，预测时自动反标准化")

# ==================== 10. 单样本预测演示 ====================
print("\n" + "=" * 60)
print("9. 单样本预测演示")
print("=" * 60)

# 新房子: 面积120, 房龄3, 房间数4
new_house = np.array([[120, 3, 4]])
new_house_scaled = scaler_X.transform(new_house)
new_house_t = torch.FloatTensor(new_house_scaled)

model.eval()
with torch.no_grad():
    pred_scaled = model(new_house_t)
    pred_price = scaler_y.inverse_transform(pred_scaled.numpy())

print(f"新房子: 面积={new_house[0][0]:.0f}, 房龄={new_house[0][1]:.0f}, 房间数={new_house[0][2]:.0f}")
print(f"预测房价: {pred_price[0][0]:.2f} 万")

# ==================== 11. 总结 ====================
print("\n" + "=" * 60)
print("10. 总结")
print("=" * 60)
print(f"✅ 模型结构: 输入层({input_size}) → 隐藏层({hidden_size}, ReLU) → 输出层({output_size})")
print(f"✅ 总参数量: {total_params:,}")
print(f"✅ 训练集 MSE: {train_mse:.2f}")
print(f"✅ 测试集 MSE: {test_mse:.2f}")
print(f"✅ 测试集 R²: {test_r2:.4f}")
print("=" * 60)