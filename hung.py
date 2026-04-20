# gpu_stress.py
import paddle
import multiprocessing as mp
import sys

def worker(gpu_id):
    paddle.set_device(f"gpu:{gpu_id}")

    # ---- 占满显存 ----
    # 获取该卡总显存（bytes）
    props = paddle.device.cuda.get_device_properties(gpu_id)
    total_mem = props.total_memory
    # 预留 12000MB 给 CUDA context / 驱动，其余全部占用
    reserve = 12000 * 1024 * 1024
    alloc_bytes = max(0, total_mem - reserve)
    n_elements = alloc_bytes // 2  # float16 = 2 bytes
    print(f"[GPU {gpu_id}] 总显存: {total_mem/1024**3:.1f} GB，"
          f"占用: {alloc_bytes/1024**3:.1f} GB", flush=True)
    mem_holder = paddle.zeros([n_elements], dtype="float16")

    # ---- 持续占满算力 ----
    N = 8192
    a = paddle.randn([N, N], dtype="float16")
    b = paddle.randn([N, N], dtype="float16")

    # 预热
    for _ in range(10):
        c = paddle.matmul(a, b)
    paddle.device.synchronize()

    # 持续计算，同时 mem_holder 保持显存占用
    while True:
        c = paddle.matmul(a, b)
        c = paddle.nn.functional.relu(c)

if __name__ == "__main__":
    mp.set_start_method("spawn")

    # 从命令行参数读取 GPU id 列表，例如：python hang.py 0 2 3
    if len(sys.argv) > 1:
        gpus = [int(x) for x in sys.argv[1:]]
    else:
        gpus = [0, 1, 2, 3]

    print(f"[hang.py] 启动占用 GPU: {gpus}", flush=True)

    procs = []
    for gid in gpus:
        p = mp.Process(target=worker, args=(gid,))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
