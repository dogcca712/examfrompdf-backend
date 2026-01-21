# 匿名用户功能实现总结

## 功能概述

实现了匿名用户（未登录用户）可以上传PDF并生成试卷的功能，同时确保：
1. 匿名用户每天只能使用1次（通过设备指纹识别）
2. 匿名用户注册后，之前的匿名使用会关联到账户并消耗免费额度
3. 免费账户可以查看历史记录（匿名用户不能）

## 实现细节

### 1. 数据库表结构

#### `guest_usage` 表
- 记录匿名用户的使用情况
- 字段：`device_fingerprint`, `ip_address`, `user_agent`, `date`, `count`
- 限制：每个设备指纹每天只能使用1次

#### `guest_to_user` 表
- 记录匿名用户到注册用户的关联
- 用于注册时关联匿名使用记录

#### `jobs` 表修改
- `user_id` 改为可空（NULL表示匿名用户）
- 新增 `device_fingerprint` 字段（用于匿名用户）

### 2. 设备指纹识别

```python
def get_device_fingerprint(request: Request) -> str:
    """生成设备指纹：IP + User-Agent"""
    ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    fingerprint_str = f"{ip}:{user_agent}"
    fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:32]
    return fingerprint
```

**特点：**
- 使用SHA256哈希，保护隐私
- 基于IP和User-Agent，难以绕过
- 32字符长度，足够唯一

### 3. API端点修改

#### `/generate` 端点
- **修改前**：必须认证（`Depends(get_current_user)`）
- **修改后**：可选认证（`Depends(get_current_user_optional)`）
- **逻辑**：
  - 认证用户：检查用量限制（按计划）
  - 匿名用户：检查设备指纹限制（每天1次）

#### `/status/{job_id}` 端点
- 支持匿名用户通过设备指纹查询
- 查询条件：`device_fingerprint = ? AND user_id IS NULL`

#### `/download/{job_id}` 端点
- 支持匿名用户通过设备指纹下载
- 查询条件：`device_fingerprint = ? AND user_id IS NULL`

#### `/auth/register` 端点
- 注册时自动关联匿名使用记录
- 如果今天有匿名使用，会消耗免费额度

### 4. 前端修改

#### `GeneratePanel.tsx`
- 修改错误处理：401错误只在已登录用户时抛出
- 无token时也可以上传（headers为空）

#### `pollJobStatus` 和 `triggerDownload`
- 已支持可选token（如果无token，headers为空）

## 使用流程

### 匿名用户流程
1. 用户未登录，直接上传PDF
2. 后端生成设备指纹（IP + User-Agent）
3. 检查今天是否已使用（`guest_usage`表）
4. 如果未使用，创建job（`user_id = NULL`, `device_fingerprint = ?`）
5. 记录匿名使用（`guest_usage`表）
6. 用户可以通过设备指纹查询状态和下载

### 注册后关联流程
1. 用户注册（`/auth/register`）
2. 后端获取设备指纹
3. 检查今天是否有匿名使用记录
4. 如果有，关联到用户账户（`guest_to_user`表）
5. 将匿名使用记录转移到用户使用记录（`usage`表）
6. 消耗免费额度

## 安全考虑

### 防止滥用
1. **设备指纹限制**：每个设备每天只能使用1次
2. **IP + User-Agent**：难以轻易绕过
3. **哈希处理**：保护隐私，不存储原始IP

### 限制
1. **单设备限制**：同一设备（IP+UA）每天只能使用1次
2. **无法查看历史**：匿名用户无法查看历史记录
3. **注册后关联**：匿名使用会消耗免费额度

## 测试建议

### 测试匿名用户上传
```bash
# 无token上传
curl -X POST https://api.examfrompdf.com/generate \
  -F "lecture_pdf=@test.pdf"

# 应该成功（第一次）
# 第二次应该返回429错误
```

### 测试设备指纹
```bash
# 同一设备第二次上传
curl -X POST https://api.examfrompdf.com/generate \
  -F "lecture_pdf=@test.pdf"

# 应该返回：Anonymous users can only generate one exam per day
```

### 测试注册关联
1. 匿名用户上传一次
2. 注册账户
3. 检查免费额度是否已消耗（应该已使用1次）

## 数据库迁移

代码已包含自动迁移：
- 添加 `device_fingerprint` 列到 `jobs` 表
- 创建 `guest_usage` 表
- 创建 `guest_to_user` 表

## 注意事项

1. **设备指纹的局限性**：
   - 同一网络（NAT）下的用户可能共享IP
   - 用户更换浏览器会生成新的指纹
   - 这是合理的限制，不是完美的

2. **注册关联时机**：
   - 只在注册时关联当天的匿名使用
   - 历史匿名使用不会关联（避免滥用）

3. **免费账户优势**：
   - 可以查看历史记录（`/jobs`端点）
   - 匿名用户无法查看历史（只能通过job_id查询）

## 未来改进

1. **更精确的设备指纹**：
   - 使用Canvas指纹
   - 使用WebGL指纹
   - 使用字体指纹

2. **更灵活的限额**：
   - 按IP限制（而非设备指纹）
   - 按时间段限制（如每小时1次）

3. **匿名用户历史**：
   - 使用localStorage存储job_id
   - 允许匿名用户查看自己的历史（通过存储的job_id）

