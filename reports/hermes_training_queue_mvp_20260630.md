# Hermes Training Queue MVP - 2026-06-30

## 目标

让用户可以通过 Hermes/飞书发送训练要求，由 Harness 自动排队、启动 MJLab、按条件停止，并顺序切换到下一个训练。

## 已完成能力

- 文本训练请求解析
- 单任务入队
- 多行批量入队
- 队列状态持久化
- Cron/Hermes monitor 自动 tick
- 无训练运行时自动启动队首任务
- 达到训练时间限制后自动暂停并切换下一个任务
- 达到 iteration 限制后自动暂停并切换下一个任务
- 启动队列任务时自动同步 `go1.yaml` 和 `feedback.yaml`
- 现有 MJLab reward 解析、L1/L2/L3、W&B 决策会跟随当前队列 run
- Git auto commit 会纳入 `runs/training_queue`
- HTTP webhook 入口 `scripts/hermes_feishu_webhook.py`
- 不启动真实训练的端到端验收脚本 `scripts/validate_hermes_feishu_e2e.py`

## 飞书/Hermes 输入示例

单个任务：

```text
G1 4096并行 8000次 1小时
```

多个任务：

```text
1. G1 4096并行 8000次 1小时
2. G1 2048并行 12000次 2小时
3. G1 4096并行 5000次 30分钟
```

全局切换间隔：

```text
1. G1 4096并行 8000次
2. G1 2048并行 12000次
3. G1 4096并行 5000次
Hermes 每3个小时自动切换到下一个训练
```

## Hermes 入口命令

飞书消息处理器可以调用：

```bash
python3 scripts/hermes_training_request.py --text "G1 4096并行 8000次 1小时"
```

批量入队也可以直接传多行文本：

```bash
python3 scripts/hermes_training_request.py --file /tmp/hermes_message.txt
```

也可以直接启动 HTTP webhook：

```bash
python3 scripts/hermes_feishu_webhook.py --host 0.0.0.0 --port 8765
```

支持：

- `GET /health`
- `GET /status`
- `POST /feishu` 或任意 POST 路径
- Feishu URL verification `challenge`
- `FEISHU_VERIFY_TOKEN` 校验

## 队列命令

查看队列：

```bash
python3 scripts/training_queue/hermes_queue.py status
```

手动 tick 一次：

```bash
python3 scripts/training_queue/hermes_queue.py tick
```

取消任务：

```bash
python3 scripts/training_queue/hermes_queue.py cancel <job_id> --reason manual_cancel
```

清理完成任务：

```bash
python3 scripts/training_queue/hermes_queue.py clear-completed
```

## 状态文件

- Queue: `runs/training_queue/queue.json`
- Events: `runs/training_queue/events.jsonl`
- Active process: `runs/active_training.json`
- Current MJLab config: `configs/tasks/mjlab/go1.yaml`
- Current feedback profile: `configs/tasks/mjlab/feedback.yaml`

## 自动停止规则

一个 job 满足任一条件就会停止：

- `max_runtime_minutes` 到达
- `mjlab/iteration + 1 >= max_iterations`
- 训练进程提前退出
- L3 反馈规则触发并暂停训练
- 用户手动取消

## 验证

```bash
python3 -m unittest tests.test_training_queue
python3 -m unittest tests.test_feishu_webhook
python3 -m py_compile scripts/training_queue/hermes_queue.py scripts/hermes_training_request.py scripts/hermes_feishu_webhook.py scripts/validate_hermes_feishu_e2e.py
bash -n scripts/run_monitor_for_hermes.sh
python3 scripts/validate_hermes_feishu_e2e.py
```

结果：通过。

## 说明

这是调度 MVP。它已经解决“飞书发训练要求 -> 自动排队 -> 自动启动/停止/切换”的核心闭环，并提供本地 HTTP webhook。生产接入时需要把飞书应用回调地址指向该 webhook 的公网/内网可达地址。更复杂的自然语言理解、鉴权签名、并发训练资源管理可以在下一轮扩展。
