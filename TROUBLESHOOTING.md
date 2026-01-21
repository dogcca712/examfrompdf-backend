# 故障排查指南

## "Failed to fetch" 错误排查

### 1. 检查认证Token
- 确保用户已登录
- 检查浏览器控制台的 Network 标签，查看请求是否包含 `Authorization: Bearer <token>` header
- 检查 localStorage 中是否有 `access_token`

### 2. 检查CORS配置
后端已配置以下允许的源：
- `https://examfrompdf.com`
- `https://www.examfrompdf.com`
- `https://examfrompdfcom.lovable.app`
- `*.lovable.app` (正则匹配)

如果前端域名不在列表中，需要添加到 `ALLOWED_ORIGINS`。

### 3. 检查后端服务状态
```bash
# 在服务器上检查
sudo systemctl status examgen
journalctl -u examgen -n 50 --no-pager
```

### 4. 检查网络连接
- 确认前端可以访问 `https://api.examfrompdf.com`
- 检查是否有防火墙阻止请求

### 5. 检查请求格式
- FormData 上传时，不要手动设置 `Content-Type` header（浏览器会自动设置）
- 确保 `Authorization` header 正确设置

### 6. 常见错误码
- **401 Unauthorized**: Token 无效或过期，需要重新登录
- **403 Forbidden**: 权限不足
- **404 Not Found**: 端点不存在或路径错误
- **500 Internal Server Error**: 服务器内部错误，查看日志

### 7. 调试步骤
1. 打开浏览器开发者工具 (F12)
2. 查看 Network 标签
3. 找到失败的请求
4. 检查：
   - Request Headers（特别是 Authorization）
   - Request Payload
   - Response Status
   - Response Body

### 8. 后端日志
查看后端日志获取详细错误信息：
```bash
journalctl -u examgen -f
```

