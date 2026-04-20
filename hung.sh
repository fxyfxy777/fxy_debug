#!/bin/bash
# hung.sh — 每小时检查 GPU 利用率，对利用率 < 50% 的卡启动 hang.py 占用

HANG_SCRIPT="/root/paddlejob/share-storage/gpfs/fanxiangyu01/hang.py"
THRESHOLD=50
CHECK_INTERVAL=3600   # 秒，1 小时

# hang.py 的后台进程 PID 文件，用于避免重复启动
PID_FILE="/tmp/hang_py.pid"

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

        # 如果 hang.py 已在运行，先杀掉旧进程
        if [ -f "$PID_FILE" ]; then
            OLD_PID=$(cat "$PID_FILE")
            if kill -0 "$OLD_PID" 2>/dev/null; then
                echo "[hung.sh] 终止旧 hang.py 进程 (PID=$OLD_PID)"
                kill "$OLD_PID"
                sleep 2
            fi
            rm -f "$PID_FILE"
        fi

        # 启动 hang.py，传入低利用率的 GPU id
        echo "[hung.sh] 启动 hang.py，目标 GPU: ${LOW_GPUS[*]}"
        nohup python "$HANG_SCRIPT" "${LOW_GPUS[@]}" >> /tmp/hang_py.log 2>&1 &
        echo $! > "$PID_FILE"
        echo "[hung.sh] hang.py 已启动，PID=$(cat $PID_FILE)"
    else
        echo "[hung.sh] 所有 GPU 利用率均 >= ${THRESHOLD}%，无需启动占用"

        # 若之前启动过 hang.py，则可选择停止（按需取消注释）
        # if [ -f "$PID_FILE" ]; then
        #     OLD_PID=$(cat "$PID_FILE")
        #     if kill -0 "$OLD_PID" 2>/dev/null; then
        #         echo "[hung.sh] GPU 利用率已恢复，终止 hang.py (PID=$OLD_PID)"
        #         kill "$OLD_PID"
        #     fi
        #     rm -f "$PID_FILE"
        # fi
    fi

    echo "[hung.sh] 等待 ${CHECK_INTERVAL}s 后再次检查..."
    sleep "$CHECK_INTERVAL"
done


# nohup bash hung.sh > hung.log 2>&1 &