# gpu_stress.py — 单卡版，由 bash 脚本循环调用
import paddle
import sys

def main(gpu_id):
    paddle.set_device(f"gpu:{gpu_id}")

    # ---- 占满显存 ----
    props = paddle.device.cuda.get_device_properties(gpu_id)
    total_mem = props.total_memory
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

    # 持续计算
    step = 0
    while True:
        c = paddle.matmul(a, b)
        c = paddle.nn.functional.relu(c)
        c = paddle.matmul(c, a)
        c = paddle.nn.functional.gelu(c)
        c = c + paddle.matmul(b, a)
        c = paddle.nn.functional.softmax(c, axis=-1)
        val = c[0, 0].item()
        step += 1
        if step % 100 == 0:
            print(f"[GPU {gpu_id}] step={step}, val={val:.6f}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"用法: python {sys.argv[0]} <gpu_id>")
        sys.exit(1)
    main(int(sys.argv[1]))
