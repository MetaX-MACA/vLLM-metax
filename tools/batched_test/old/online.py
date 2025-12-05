# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import time
import requests
import os
import socket
import psutil
import sys
import atexit
import signal
import json
from threading import Thread

# ----------------------------
# 标准化测试 PROMPTS
# ----------------------------
TEST_PROMPTS = [
    ("中国的首都是哪？请用中文回答。", "北京"),
    ("苹果公司的创始人是谁？请用中文回答。", "乔布斯"),
    ("世界上哪个国家的国土面积最大？请用中文回答。", "俄罗斯"),
    ("太阳从哪边升起？请用中文回答。", "东"),
]


# ----------------------------
# GPU unlock 函数
# ----------------------------
def unlock_gpus_py(gpus, log_file=None):
    if gpus:
        lock_dir = "/tmp/gpu_locks"
        for g in gpus.split(","):
            if g:
                lock_file = f"{lock_dir}/gpu{g}.lock"
                try:
                    os.remove(lock_file)
                    msg = f"[UNLOCK] GPU {g} 锁已释放"
                    print(msg)
                    if log_file:
                        with open(log_file, "a") as f:
                            f.write(msg + "\n")
                except FileNotFoundError:
                    pass


# ----------------------------
# 启动 vLLM serve
# ----------------------------
def start_server(args, log_file):
    if not os.path.exists(args.model_path):
        msg = f"[ERROR] 模型路径不存在: {args.model_path}"
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
        unlock_gpus_py(args.gpus, log_file)
        sys.exit(1)

    if (
        getattr(args, "decode_context_parallel_size", 1)
        and args.decode_context_parallel_size > 1
    ):
        if args.tensor_parallel_size % args.decode_context_parallel_size != 0:
            msg = f"[ERROR] tp={args.tensor_parallel_size} 不能被 dcp={args.decode_context_parallel_size} 整除，终止启动"
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
            unlock_gpus_py(args.gpus, log_file)
            sys.exit(1)

    cmd = [
        "vllm",
        "serve",
        args.model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--pipeline-parallel-size",
        str(args.pipeline_parallel_size),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--distributed-executor-backend",
        args.distributed_executor_backend,
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--swap-space",
        "16",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]

    if getattr(args, "decode_context_parallel_size", None):
        cmd += [
            "--decode-context-parallel-size",
            str(args.decode_context_parallel_size),
        ]

    if args.speculative_config:
        cmd += ["--speculative-config", args.speculative_config]
    if args.hf_overrides:
        cmd += ["--hf-overrides", args.hf_overrides]
    if args.extra_args:
        cmd += args.extra_args

    try:
        proc = subprocess.Popen(cmd, env=os.environ.copy())
        return proc
    except Exception as e:
        msg = f"[ERROR] 启动模型失败: {e}"
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
        unlock_gpus_py(args.gpus, log_file)
        sys.exit(1)


# ----------------------------
# 异步等待端口
# ----------------------------
def wait_for_port_async(host, port, timeout, log_file):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=5):
                msg = f"[INFO] 端口 {port} 已开放"
                print(msg)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                return True
        except OSError:
            elapsed = int(time.time() - start)
            msg = f"[WAIT] 端口 {port} 未开放，已等待 {elapsed} 秒..."
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
            time.sleep(5)
    msg = f"[ERROR] 等待端口 {port} 超时"
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")
    return False


# ----------------------------
# 测试模型请求（修改版，增加精度判断）
# ----------------------------
def test_server(port, model_path, log_file):
    base_url = f"http://127.0.0.1:{port}"

    for question, keyword in TEST_PROMPTS:
        precision_success = False
        text = ""
        # 1) chat 测试
        url = f"{base_url}/v1/chat/completions"
        payload = {
            "model": model_path,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 100,
            "temperature": 0.95,
            "top_p": 0.95,
        }

        try:
            r = requests.post(url, json=payload, timeout=90)
            if r.status_code == 200:
                answer = r.json()
                text = (
                    answer.get("choices", [{}])[0].get("message", {}).get("content")
                    or answer.get("choices", [{}])[0].get("text")
                    or str(answer)
                )

                if keyword in text:
                    precision_success = True
                    msg = f"[OK] (chat) {question!r} -> {text}"
                else:
                    msg = f"[FAIL] (chat) 回答不包含关键字「{keyword}」: {text}"

            else:
                msg = f"[FAIL] (chat) {r.status_code}: {r.text}"

        except Exception as e:
            msg = f"[ERROR] (chat) {e}"

        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
        # 2) completions fallback
        if not precision_success:
            url = f"{base_url}/v1/completions"
            payload = {
                "model": model_path,
                "prompt": question,
                "max_tokens": 100,
                "temperature": 0.95,
                "top_p": 0.95,
            }

            try:
                r = requests.post(url, json=payload, timeout=90)
                if r.status_code == 200:
                    answer = r.json()
                    text = (
                        answer.get("choices", [{}])[0].get("text")
                        or answer.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content")
                        or str(answer)
                    )

                    if keyword in text:
                        precision_success = True
                        msg = f"[OK] (completions) {question!r} -> {text}"
                    else:
                        msg = f"[FAIL] (completions) 回答不包含关键字「{keyword}」: {text}"

                else:
                    msg = f"[FAIL] (completions) {r.status_code}: {r.text}"

            except Exception as e:
                msg = f"[ERROR] (completions) {e}"

            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
        # 3) 最终精度输出
        if precision_success:
            msg = f"✅ [PRECISION] 模型回答包含关键字「{keyword}」"
        else:
            msg = f"❌ [PRECISION] 模型回答未包含关键字「{keyword}」"

        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")


# ----------------------------
# 杀死进程树
# ----------------------------
def kill_process_tree(proc, log_file=None, timeout=30):
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
        msg = f"[INFO] 进程 {proc.pid} 已优雅退出"
        print(msg)
        if log_file:
            open(log_file, "a").write(msg + "\n")
    except subprocess.TimeoutExpired:
        msg = f"[WARN] 进程 {proc.pid} 超时未退出，强制 kill"
        print(msg)
        if log_file:
            open(log_file, "a").write(msg + "\n")
        try:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                msg = f"[KILL] 子进程 {child.pid}"
                print(msg)
                if log_file:
                    open(log_file, "a").write(msg + "\n")
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        proc.wait()
        msg = f"[INFO] 进程 {proc.pid} 已被强制清理"
        print(msg)
        if log_file:
            open(log_file, "a").write(msg + "\n")


# ----------------------------
# main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--distributed_executor_backend", default="ray")
    parser.add_argument("--speculative-config", type=str)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--hf-overrides", type=str)
    parser.add_argument("--gpus", type=str)
    parser.add_argument("--decode-context-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    parser.add_argument("--log_file", type=str, required=True)
    args = parser.parse_args()

    if args.hf_overrides:
        try:
            hf_overrides_dict = json.loads(args.hf_overrides)
            print(f"Loaded hf_overrides: {hf_overrides_dict}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] 无法解析 hf_overrides 参数: {e}")
            hf_overrides_dict = {}
    else:
        hf_overrides_dict = {}

    log_file = args.log_file

    proc = None

    def exit_handler(signum=None, frame=None):
        if proc:
            kill_process_tree(proc, log_file)
        unlock_gpus_py(args.gpus, log_file)
        sys.exit(0)

    atexit.register(exit_handler)
    signal.signal(signal.SIGINT, exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)

    try:
        proc = start_server(args, log_file)
        port_ok = wait_for_port_async(
            "127.0.0.1", args.port, timeout=3600, log_file=log_file
        )
        if not port_ok:
            unlock_gpus_py(args.gpus, log_file)
            kill_process_tree(proc, log_file)
            sys.exit(1)

        t = Thread(target=test_server, args=(args.port, args.model_path, log_file))
        t.start()
        t.join()
    except Exception as e:
        msg = f"[EXCEPTION] 模型 {args.model_path} 测试过程中发生错误: {e}"
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
        unlock_gpus_py(args.gpus, log_file)
        if proc:
            kill_process_tree(proc, log_file)
        sys.exit(1)
    finally:
        if proc:
            kill_process_tree(proc, log_file)
        unlock_gpus_py(args.gpus, log_file)


if __name__ == "__main__":
    main()
