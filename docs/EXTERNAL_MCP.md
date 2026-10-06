# 外部 MCP 服务

Studio 的本地项目操作使用原生工具，MCP 客户端仅接入用户配置的外部 stdio / Streamable HTTP 服务。
可在「设置 → MCP 服务管理」添加、编辑、启用/禁用、删除服务，并确认后测试连接、查看工具列表。
界面保存到同一个配置文件，不自动连接服务；认证请求头填写环境变量名，不回显已有 stdio 私有环境变量。
暂无 OAuth 登录流程或旧 HTTP SSE 适配器。

默认配置位置为 `%APPDATA%\RAgent\mcp_servers.json`。也可在启动 RAgent 前设置
`RAGENT_MCP_CONFIG` 为配置文件的绝对路径。不会自动读取项目目录中的 MCP 配置。
EXE 使用时，外部服务需要的 Python、Node、可执行程序等仍需自行安装；打包不携带这些外部依赖。

```json
{
  "mcpServers": {
    "local_service": {
      "transport": "stdio",
      "command": "C:\\Program Files\\nodejs\\node.exe",
      "args": ["C:\\my-mcp\\server.js"],
      "cwd": "C:\\my-mcp"
    },
    "remote_service": {
      "transport": "streamable_http",
      "url": "https://your-server.example/mcp",
      "headers_env": {"Authorization": "MY_MCP_AUTHORIZATION"}
    }
  }
}
```

这是配置结构示例，不是可直接使用的真实服务。`headers_env` 的值是环境变量名；
如使用 Bearer 认证，该变量的内容应为 `Bearer ...`。不要把真实凭据提交到 Git。
stdio 可用 `env` 映射设置子进程环境变量；优先使用服务所支持的安全凭据管理方式。
非本机 HTTP 必须使用 HTTPS。配置加载和结果解析不是外部服务的沙箱。

模型工具调用方式：

- `mcp_tool="list_servers"`：只列出服务名称和传输方式，不连接外部服务。
- `mcp_tool="local_service::list_tools"`：启动并发现指定服务的工具。
- `mcp_tool="local_service::实际工具名"`：使用 `mcp_arguments` 提供参数。
- 不再提供内置 MCP；无前缀工具和 `builtin::工具名` 不再可用。项目文件使用原生 `list_files/read/search`。
- 桌面发布只需要 `RAgent.exe`，不再附带 `RAgent-MCP.exe`。

外部工具发现也可能启动代码/发起网络连接，因此默认需要审批。
完全授权模式或匹配的明确授权可放行；工具的 readOnly 注释不用于自动免审批。
授权指纹绑定服务配置，修改命令、地址或配置后需要重新授权。
批准外部服务不等于服务被沙箱隔离；只配置你信任的程序和地址。

客户端每次调用建立新连接，初始化、分页读取工具列表、调用，再关闭连接。
外部结果保留 `content`、`structuredContent` 和错误信息，不要求文本必须是 JSON。
图片等内容保留为协议数据，不保证聊天界面直接渲染所有媒体。
总调用上限为 30 秒，当前不支持长时间 MCP 任务或持久会话。
