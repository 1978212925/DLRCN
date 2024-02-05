
# 结节初始学习率0.00001,没有用迁移学习
# 脂肪初始学习率0.00000001，迁移学习
cfg = dict(
    model_name='',
    num_classes=2,
    lr_start=0.000003,
    weight_decay=5e-2,
    epochs=500,
    train_batch=12,
    val_batch=12,
    class_weights=[8, 1],
    respth='Running_Dict',
)
