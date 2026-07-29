#!/bin/bash
# 自动获取 lshrun.py 的绝对路径并写入 ~/.bashrc

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LSHRUN_PY="${SCRIPT_DIR}/lshrun.py"

if [ ! -f "$LSHRUN_PY" ]; then
    echo "Error: lshrun.py not found at $LSHRUN_PY"
    exit 1
fi

# 检查是否已经配置过
if grep -q "alias lshrun=" ~/.bashrc 2>/dev/null; then
    echo "lshrun alias already exists in ~/.bashrc, updating..."
    sed -i "s|alias lshrun=.*|alias lshrun=\"python ${LSHRUN_PY}\"|" ~/.bashrc
else
    cat >> ~/.bashrc <<EOF

# ===== lshrun alias =====
alias lshrun="python ${LSHRUN_PY}"
# ===== End lshrun alias =====
EOF
fi

# 当前 shell 立即生效
alias lshrun="python ${LSHRUN_PY}"

echo "Done. lshrun -> python ${LSHRUN_PY}"
echo "Run 'source ~/.bashrc' or open a new terminal to use lshrun globally."
