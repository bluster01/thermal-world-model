"""
config.py — 世界模型实验全局配置
基于 Exp-0 config_improved.py，改造为 (s,a)→s' 世界模型范式
"""

# ===== 数据 =====
DATA_DIR = "data/伊敏6号机/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT"
TRAIN_FILE = "A侧主汽温全数据03_cleaned_10s.csv"

# ===== 序列 =====
WINDOW_SIZE = 96      # 历史窗口 (16 min @ 10s)
PRED_LEN = 1          # 预测步长 (世界模型: 单步)
ROLLOUT_LEN = 18      # 展开步数 (训练和评估用)

# ===== 特征 (v2 clean: 去掉设定值+阀位, 保留11维纯物理状态) =====
TARGET_IDX = 10      # 末级过热器出口汽温在11维中的索引

# 11维纯物理状态 (去掉 '二级减温中间设定值', 阀位不再作为状态)
FEATURE_COLUMNS = [
    '机组负荷',                    # 0
    '主蒸汽压力',                  # 1
    '机组负荷变化率',              # 2
    '主蒸汽流量',                  # 3
    '未校正总煤量',                # 4
    '分离器出口温度',              # 5
    '一级减温器入口温度',          # 6
    '二级减温器入口温度',          # 7
    '一级减温器出口温度',          # 8
    '二级减温器出口温度',          # 9
    '末级过热器出口汽温',          # 10 ← TARGET
]

# 动作定义: 差分阀位 a_t = [Δv1_t, Δv2_t] = [v_t - v_{t-1}]
# 在数据加载时从绝对值计算
VALVE_ABS_COLS = ['一级减温调节门阀位', '二级减温调节门阀位']
N_STATE = 11
N_ACTION = 2

# ===== 模型 (继承 Exp-0 TCN-iTransformer) =====
D_MODEL = 64
N_HEADS = 4
N_VAR_LAYERS = 2
N_TCN_LAYERS = 2
PATCH_LEN = 16
STRIDE = 8
DROPOUT = 0.1

# ===== 训练 =====
BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20  # 宽松早停: 给足训练时长 (exp_025b 审查: rollout 最优在 E10 之后)
WEIGHT_DECAY = 1e-5
ROLLOUT_LOSS_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.2]  # 前5步权重递减

# ===== 路径 =====
MODEL_SAVE_DIR = "results/models"
LOG_DIR = "results/logs"

# ===== 设备 =====
DEVICE = "cuda"
