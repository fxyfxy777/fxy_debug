---
name: setup-env-fxy
description: 在新机器上一键配置 fxy 开发环境，包括 proxy 函数（agent.baidu.com:8891）和 lshrun 工具 alias。当用户说"配置环境"、"新机器配置"、"setup proxy"、"setup lshrun"、"配置 proxy 和 lshrun"、"换台机器配置"、"帮我配置开发环境"、"setup-env-fxy" 时触发。
---

# setup-env-fxy

在新机器上配置 fxy 的常用开发工具：proxy 函数 + lshrun alias。

## 脚本位置

代码仓库：`/root/paddlejob/inference-public/fanxiangyu/fxy_debug/`

- **proxy 配置脚本**：`proxy.sh`
- **lshrun 配置脚本**：`setup_lshrun.sh`

## 执行步骤

1. 确认 fxy_debug 目录存在：
   ```bash
   ls /root/paddlejob/inference-public/fanxiangyu/fxy_debug/
   ```

2. 执行 proxy 配置（将 proxy 函数写入 ~/.bashrc）：
   ```bash
   bash /root/paddlejob/inference-public/fanxiangyu/fxy_debug/proxy.sh
   ```

3. 执行 lshrun 配置（将 alias 写入 ~/.bashrc）：
   ```bash
   bash /root/paddlejob/inference-public/fanxiangyu/fxy_debug/setup_lshrun.sh
   ```

4. 验证配置写入成功：
   ```bash
   grep -A 5 "Proxy helper" ~/.bashrc
   grep "lshrun" ~/.bashrc
   ```

5. 让配置在当前 shell 生效：
   ```bash
   source ~/.bashrc
   ```

## 配置效果

**proxy 函数** — 写入 `~/.bashrc` 后，可用以下命令：
- `proxy` 或 `proxy on` — 开启代理（host: agent.baidu.com, port: 8891）
- `proxy off` — 关闭代理
- `proxy status` — 查看当前代理状态
- `proxy <端口>` — 使用自定义端口

**lshrun alias** — 写入 `~/.bashrc` 后：
- `lshrun` 等价于 `python /root/paddlejob/inference-public/fanxiangyu/fxy_debug/lshrun.py`

## 注意事项

- `proxy.sh` 会追加到 `~/.bashrc`，不会重复添加（有幂等保护）
- `setup_lshrun.sh` 已有幂等保护：若 alias 已存在则更新，不存在则新增
- 配置完成后需要 `source ~/.bashrc` 或重开终端才能在当前 shell 生效
- lshrun alias 中的路径是绝对路径，换机器时如果仓库路径不同需要重新运行 `setup_lshrun.sh`
