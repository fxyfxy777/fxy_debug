#!/usr/bin/env python
"""持续运行 matmul 并在 for 循环中统计计算速度 (TFLOPS)，支持多卡并行"""
import torch
import torch.multiprocessing as mp
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="GPU ID 列表，逗号分隔")
parser.add_argument("--n", type=int, default=8192, help="矩阵维度 NxN")
parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
parser.add_argument("--iters", type=int, default=100, help="每轮迭代次数")
args = parser.parse_args()

gpu_ids = [int(x) for x in args.gpus.split(",")]


def worker(gpu_id, n, dtype_str, iters):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[dtype_str]

    print(f"[GPU {gpu_id}] 启动: {torch.cuda.get_device_name(gpu_id)}, {n}x{n} {dtype_str}", flush=True)

    a = torch.randn(n, n, dtype=dtype, device=device)
    b = torch.randn(n, n, dtype=dtype, device=device)

    # 预热
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()

    flops_per_matmul = 2.0 * n * n * n
    round_idx = 0
    while True:
        round_idx += 1
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(iters):
            c = torch.matmul(a, b)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        tflops = flops_per_matmul * iters / elapsed / 1e12
        print(f"[GPU {gpu_id}][Round {round_idx:>4}] {elapsed:.3f}s | "
              f"{tflops:.2f} TFLOPS | {elapsed/iters*1000:.2f} ms/iter", flush=True)


if __name__ == "__main__":
    print(f"启动 {len(gpu_ids)} 张卡: {gpu_ids}")
    mp.set_start_method("spawn", force=True)
    procs = []
    for gpu_id in gpu_ids:
        p = mp.Process(target=worker, args=(gpu_id, args.n, args.dtype, args.iters))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
