# Hermes Training Queue Product Completion - 2026-07-01

## 结论

用户通过飞书/Hermes 下发 MJLab 训练要求的产品闭环已经补齐到可验收 MVP：

- 飞书/Hermes HTTP 消息入口已实现。
- 单个训练要求可入队。
- 多个训练要求可按顺序入队。
- 队列 tick 可自动启动队首任务。
- 达到训练时间限制可自动暂停当前训练。
- 达到 iteration 限制可自动暂停当前训练。
- 当前任务结束后可自动切换到下一个训练。
- Hermes monitor 定时流程已接入队列 tick。
- L1/L2/L3、MJLab reward 解析、W&B 决策会跟随队列同步后的当前 run。
- Git auto commit 已覆盖 `runs/training_queue`。

## 新增文件

- `scripts/hermes_feishu_webhook.py`
- `scripts/start_hermes_feishu_webhook.sh`
- `scripts/stop_hermes_feishu_webhook.sh`
- `scripts/validate_hermes_feishu_e2e.py`
- `tests/test_feishu_webhook.py`
- `reports/hermes_feishu_e2e_validation_20260701.md`

## 飞书接入方式

启动 webhook：

```bash
bash scripts/start_hermes_feishu_webhook.sh
```

如果飞书应用配置了 verification token：

```bash
FEISHU_VERIFY_TOKEN=<token> bash scripts/start_hermes_feishu_webhook.sh
```

飞书后台将消息事件回调到该服务后，服务会从 `event.message.content` 中提取文本并写入训练队列。

## 验收结果

```bash
python3 -m unittest tests.test_training_queue tests.test_feishu_webhook
python3 -m py_compile scripts/training_queue/hermes_queue.py scripts/hermes_training_request.py scripts/hermes_feishu_webhook.py scripts/validate_hermes_feishu_e2e.py
bash -n scripts/start_hermes_feishu_webhook.sh scripts/stop_hermes_feishu_webhook.sh scripts/run_monitor_for_hermes.sh
python3 scripts/validate_hermes_feishu_e2e.py
```

结果：

- 21 个队列、webhook、MJLab feedback 相关单测通过。
- Python 编译检查通过。
- Shell 语法检查通过。
- 模拟飞书消息端到端验收通过。
- 本机 HTTP webhook `/health` 和 Feishu `challenge` 回调验证通过。

## 尚未覆盖

这次没有启动真实 4096 并行训练，也没有配置真实飞书公网回调地址。代码侧已经具备入口；上线侧需要把飞书应用回调 URL 指向运行中的 `hermes_feishu_webhook.py` 服务。
