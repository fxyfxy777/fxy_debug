#!/bin/bash
# hung.sh — 每小时检查 GPU 利用率，对利用率 < 50% 的卡启动 hung.py 占用

HANG_SCRIPT="hung.py"
THRESHOLD=50
CHECK_INTERVAL=3600   # 秒，1 小时

# hung.py 的后台进程 PID 文件前缀
PID_DIR="/tmp/hang_py_pids"

# 清理旧的 hung.py 进程
cleanup_old() {
    if [ -d "$PID_DIR" ]; then
        for pid_file in "$PID_DIR"/*.pid; do
            if [ -f "$pid_file" ]; then
                OLD_PID=$(cat "$pid_file")
                if kill -0 "$OLD_PID" 2>/dev/null; then
                    echo "[hung.sh] 终止旧 hung.py 进程 (PID=$OLD_PID)"
                    kill "$OLD_PID"
                fi
                rm -f "$pid_file"
            fi
        done
    fi
}

echo "[hung.sh] 启动 GPU 利用率监控，阈值=${THRESHOLD}%，检查间隔=${CHECK_INTERVAL}s"

while true; do
    # 获取每张卡的利用率：格式为 "gpu_id,utilization"
    mapfile -t GPU_INFO < <(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')

    LOW_GPUS=()
    for entry in "${GPU_INFO[@]}"; do
        gpu_id="${entry%%,*}"
        util="${entry##*,}"
        echo "[hung.sh] GPU $gpu_id 利用率: ${util}%"
        if [ "$util" -lt "$THRESHOLD" ]; then
            LOW_GPUS+=("$gpu_id")
        fi
    done

    if [ "${#LOW_GPUS[@]}" -gt 0 ]; then
        echo "[hung.sh] 以下 GPU 利用率低于 ${THRESHOLD}%: ${LOW_GPUS[*]}"

        # 先清理旧进程
        cleanup_old

        # 创建 PID 目录
        mkdir -p "$PID_DIR"

        # 为每个低利用率 GPU 启动独立的 hung.py 进程
        for gpu_id in "${LOW_GPUS[@]}"; do
            echo "[hung.sh] 启动 hung.py，目标 GPU: $gpu_id"
            nohup python "$HANG_SCRIPT" "$gpu_id" >> "/tmp/hang_py_${gpu_id}.log" 2>&1 &
            PID=$!
            echo "$PID" > "$PID_DIR/hang_py_${gpu_id}.pid"
            echo "[hung.sh] GPU $gpu_id 进程已启动，PID=$PID"
        done
    else
        echo "[hung.sh] 所有 GPU 利用率均 >= ${THRESHOLD}%，清理占用进程"
        # 清理所有 hung.py 进程
        cleanup_old
    fi

    echo "[hung.sh] 等待 ${CHECK_INTERVAL}s 后再次检查..."
    sleep "$CHECK_INTERVAL"
done


# nohup bash run.sh > hung.log 2>&1 &
# ps -eo pid,cmd | grep "python hung.py" | grep -v grep | awk '{print $1}' | xargs kill -9