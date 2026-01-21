# 服务器检查清单

## 当 POST /generate 失败时

### 1. 检查服务状态
```bash
sudo systemctl status examgen
```

**期望输出：**
- `Active: active (running)`
- 没有错误信息

**如果服务停止：**
```bash
sudo systemctl restart examgen
sudo systemctl status examgen
```

### 2. 查看最近日志
```bash
journalctl -u examgen -n 100 --no-pager
```

**查找：**
- 错误堆栈（Traceback）
- "Failed to" 相关错误
- 数据库错误
- 文件系统错误

### 3. 检查磁盘空间
```bash
df -h
```

**确保：**
- `/home/ubuntu/examgen` 所在分区有足够空间
- 至少 1GB 可用空间

### 4. 检查文件权限
```bash
ls -la /home/ubuntu/examgen/
ls -la /home/ubuntu/examgen/build_jobs/
```

**确保：**
- `build_jobs` 目录可写
- `data.db` 文件可读写

### 5. 检查 Python 依赖
```bash
cd /home/ubuntu/examgen
source .venv/bin/activate
pip list | grep -E "fastapi|uvicorn|jwt|bcrypt|stripe"
```

**确保所有依赖已安装：**
- fastapi
- uvicorn
- PyJWT
- bcrypt
- stripe

### 6. 测试后端健康检查
```bash
curl -I http://127.0.0.1:8000/health
```

**期望：** `HTTP/1.1 200 OK`

### 7. 检查 Nginx 配置
```bash
sudo nginx -t
sudo systemctl status nginx
```

### 8. 查看实时日志
```bash
journalctl -u examgen -f
```

然后在前端尝试上传，观察日志输出。

### 9. 常见问题

#### 问题：服务启动失败
```bash
# 查看详细错误
journalctl -u examgen -n 200 --no-pager

# 手动测试启动
cd /home/ubuntu/examgen
source .venv/bin/activate
python -c "import app; print('Import OK')"
```

#### 问题：数据库锁定
```bash
# 检查是否有其他进程在使用数据库
lsof /home/ubuntu/examgen/data.db

# 如果数据库损坏，备份后重建
cp /home/ubuntu/examgen/data.db /home/ubuntu/examgen/data.db.backup
```

#### 问题：内存不足
```bash
# 检查内存使用
free -h

# 检查是否有 OOM killer
dmesg | grep -i "out of memory"
```

### 10. 重启服务
```bash
sudo systemctl restart examgen
sleep 2
sudo systemctl status examgen
```

### 11. 验证修复
```bash
# 测试健康检查
curl http://127.0.0.1:8000/health

# 测试认证（需要token）
curl -H "Authorization: Bearer YOUR_TOKEN" http://127.0.0.1:8000/auth/me
```

## 日志分析

### 关键日志模式

**成功创建任务：**
```
Job created: <job_id> for user <user_id>, file: <filename>
Job <job_id> created in database
File saved: <filename>, size: X.XXMB
Background job thread started for <job_id>
```

**错误模式：**
- `Database error:` - 数据库问题
- `Failed to save file:` - 文件保存失败
- `Failed to create job:` - 任务创建失败
- `Traceback` - Python 异常

## 紧急恢复

如果服务完全无法启动：

```bash
# 1. 停止服务
sudo systemctl stop examgen

# 2. 备份当前状态
cd /home/ubuntu/examgen
cp data.db data.db.backup.$(date +%Y%m%d_%H%M%S)

# 3. 检查代码
git status
git log --oneline -5

# 4. 拉取最新代码
git pull origin main

# 5. 重启服务
sudo systemctl start examgen
sudo systemctl status examgen
```

