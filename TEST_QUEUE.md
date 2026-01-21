# 测试队列功能指南

## 方法1：使用公开端点（推荐，无需认证）

```bash
# 直接访问公开端点（无需token）
curl https://api.examfrompdf.com/queue/status/public
```

**响应示例：**
```json
{
  "queue_size": 3,
  "max_queue_size": 10,
  "processing": 2,
  "max_processing": 5,
  "max_concurrent": 5,
  "total_processed": 50,
  "total_failed": 1,
  "estimated_wait_time": 72
}
```

## 方法2：使用认证端点（需要token）

### 步骤1：获取Token
```bash
# 登录获取token
curl -X POST https://api.examfrompdf.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "your@email.com", "password": "yourpassword"}'

# 响应：
# {
#   "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
#   "user": {...}
# }
```

### 步骤2：使用Token查询队列状态
```bash
# 使用获取的token
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  https://api.examfrompdf.com/queue/status
```

## 方法3：在服务器上直接测试（本地）

```bash
# 在服务器上测试（无需通过Nginx）
curl http://127.0.0.1:8000/queue/status/public
```

## 方法4：查看日志监控

```bash
# 查看队列相关日志
journalctl -u examgen -f | grep -E "Queue size|queued|Worker thread"

# 示例输出：
# Job abc123 queued. Queue size: 5, Processing: 3/5
# Worker thread: Starting job abc123
# Worker thread: Completed job abc123
```

## 测试并发限制

### 测试1：同时发送多个请求
```bash
# 同时发送10个请求
for i in {1..10}; do
  curl -X POST https://api.examfrompdf.com/generate \
    -H "Authorization: Bearer YOUR_TOKEN" \
    -F "lecture_pdf=@test.pdf" &
done

# 然后查看队列状态
curl https://api.examfrompdf.com/queue/status/public

# 应该看到：
# - processing: 最多5个（MAX_CONCURRENT_JOBS）
# - queue_size: 其他任务在排队
```

### 测试2：观察队列变化
```bash
# 持续监控队列状态
watch -n 2 'curl -s https://api.examfrompdf.com/queue/status/public | jq'
```

## 关键指标说明

- **queue_size**: 当前排队数（等待处理的任务数）
- **max_queue_size**: 历史最高排队数 ⭐（回答你的问题）
- **processing**: 当前正在处理的任务数
- **max_processing**: 历史最高同时处理数 ⭐
- **max_concurrent**: 最大并发数（配置值）
- **estimated_wait_time**: 预估等待时间（秒）

## 验证并发限制是否生效

1. 同时发送10个PDF上传请求
2. 立即查看队列状态
3. 应该看到：
   - `processing` ≤ `max_concurrent` (5)
   - `queue_size` = 10 - `processing`
   - `max_queue_size` 会记录历史最高值

## 故障排查

如果队列状态显示异常：
1. 检查服务是否运行：`sudo systemctl status examgen`
2. 查看日志：`journalctl -u examgen -n 50`
3. 检查代码是否更新：`git log --oneline -1`

