# Mac 写代码 Windows 编译

在 Mac 上用 Claude 写代码、在 Windows 上编译（比如 Vivado、MSVC），让 Claude 自己完成"改代码 → 编译 → 看报错 → 再改"的闭环。

## 三种做法

| 做法 | 方式 | 适合 |
|---|---|---|
| **A. 代码在 Mac，远程编译** | 脚本先用 rsync 同步代码，再用 SSH 在 Win 上执行编译，输出返回 Mac | 代码跨平台、想在 Mac 上管理代码 |
| **B. VS Code Remote-SSH 连 Win** | VS Code 连到 Win，Claude Code 扩展运行在 Win 一侧，直接读写 Win 上的文件、执行命令 | 纯 Windows 项目，需要频繁运行和调试 |
| **C. Win 上直接装 Claude Code** | 效果同 B，但要在 Win 上操作 | 不需要坐在 Mac 前时 |

目前选定的是 **B**：Mac 上的 VS Code 通过 Remote-SSH 打开 Win 上的项目，Claude 在 Win 一侧干活。

## B 方案注意事项

1. **Git for Windows**：Claude Code 在 Windows 上借助 Git Bash 执行命令，需要先装
2. **登录**：第一次在远程端使用要登录，授权链接在 Mac 的浏览器里打开即可
3. **编译环境变量**：`cl`、`msbuild` 等通常只在 Developer Command Prompt 里可用，写一个编译脚本统一处理：
   ```bat
   :: build.bat
   call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
   msbuild Proj.sln /p:Configuration=Debug /v:minimal
   ```
   并在项目的 `CLAUDE.md` 里写明"编译用 build.bat"
4. **SSH 默认 shell** 是 cmd，可改成 PowerShell
5. **GUI 程序**：通过 SSH 启动的程序看不到窗口；命令行程序和测试不受影响

## 让 Claude 能自主闭环的关键

1. 一条命令就能编译，输出打到终端，不弹窗、不交互
2. 最好有一条命令能跑测试，验证改动正确，而不只是能编译
3. 出错时返回非 0 退出码

## Vivado 与 MCP（查证于 2026-11-15）

- **AMD Ross**（2026-09-30 发布）：AMD 官方的智能体式 AI 助手，覆盖 Vivado 和 Vitis HLS，提供 Windows/Linux 版的 Vivado MCP 服务器（2026.9.1），支持 Claude Code。自带 20 多个 Agent Skills，不需要额外许可证，内置人工审核关口。文档在 GitHub `xilinx/ross-ai-assistant`
- **社区方案**：vivado-mcp-native（跨平台）、vivado-mcp（通过 SSH 跑远程 Linux 上的 Vivado）、vivado-mcp-agent（面向 Windows）、SynthPilot（需要许可证）
- **注意**：完整实现流程可能要几十分钟，要确认方案支持后台任务或较长超时
- **对简单项目来说**，用 `vivado -mode batch -source build.tcl` 批处理加 SSH 就够了，不一定需要 MCP

以上方案来自搜索结果，尚未实际安装验证。

## 相关
- [FPGA 矩阵运算通讯全流程](01-总体设计.md)
