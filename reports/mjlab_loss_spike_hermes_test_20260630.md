# MJLab Loss Spike Hermes Test - 2026-06-30

## 目标

验证 MJLab 指标中的 loss 被人为放大后，Hermes agent 是否能自动发现异常。

## 处理方式

没有移动 `/home/leju/mjlab/src`，也没有修改 MJLab 源码。Harness 已经能从 MJLab 日志解析出 `train/value_loss`，所以本次使用隔离的临时指标文件测试 Hermes 发现能力：

- 临时 profile: `/tmp/mjlab_loss_spike_hermes_test/feedback.yaml`
- 临时指标: `/tmp/mjlab_loss_spike_hermes_test/train.jsonl`
- 正式规则: `configs/tasks/mjlab/detection_rules.yaml`

## 新增规则

在 `configs/tasks/mjlab/detection_rules.yaml` 增加：

- `mjlab_g1_value_loss_explosion_l3`
- metric: `train/value_loss`
- condition: `train/value_loss > 1000`
- response_level: `L3`
- emergency_action: `pause_training`

## 测试数据

写入临时 `train.jsonl`：

- step 1000: `train/value_loss = 0.02`
- step 2000: `train/value_loss = 0.03`
- step 3000: `train/value_loss = 50000.0`

## 执行命令

```bash
python3 scripts/feedback/monitor_hermes.py --config /tmp/mjlab_loss_spike_hermes_test/feedback.yaml --debug
```

## 结果

Hermes 成功触发 L3：

```text
规则：mjlab_g1_value_loss_explosion_l3
级别：L3
指标：train/value_loss
最新 step：3000
最新值：50000.000
触发原因：当前值 50000.000 > 阈值 1000.000
紧急动作：pause_training
```

## 验证

```bash
python3 scripts/ops/lint_config.py
python3 -m unittest tests.test_mjlab_feedback_control
python3 -m py_compile tests/test_mjlab_feedback_control.py scripts/feedback/monitor_hermes.py scripts/feedback/l3_check.py
```

结果：

- 机械化校验通过
- `tests.test_mjlab_feedback_control`: 9 tests OK
- py_compile 通过

## 结论

完成。MJLab loss 被放大后，Hermes agent 已能通过 `train/value_loss` 自动发现异常并升级为 L3 告警。

本次没有执行真实 `pause_training`，避免误影响当前 MJLab 状态；只验证了 Hermes 的发现与告警链路。L3 真实暂停能力已经在前序 MJLab 4096 测试中验证过。
