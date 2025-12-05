#!/usr/bin/env bash
set -euo pipefail
set +m
########################################
# model_list_test.sh (最终强化版，支持 dcp)
########################################
ray stop
ulimit -n 65536

export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export MACA_DIRECT_DISPATCH=1
unset CUDA_VISIBLE_DEVICES 

export -p

MODELS_JSON="models.json"
BASE_LOG_DIR="./logs/online_test"
LOCK_DIR="/tmp/gpu_locks"
ALLOC_LOCK="/tmp/gpu_alloc.lock"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
RUN_LOG_DIR="${BASE_LOG_DIR}/${TIMESTAMP}"
FAILED_FILE="${RUN_LOG_DIR}/failed_models.txt"

mkdir -p "$RUN_LOG_DIR" "$LOCK_DIR"
rm -f "$FAILED_FILE"

TOTAL_GPUS=8
PORT_BASE=23500
WAIT_INTERVAL=60
CHECK_INTERVAL=3

declare -A pid_to_gpus
declare -A pid_to_log
declare -A pid_to_model
declare -a launched_pids

##############################################
# 解析 models.json
##############################################
models_lines=$(python3 - <<'PY'
import json
cfg = json.load(open("models.json"))
for m in cfg:
    model = m.get("model","")
    tp = int(m.get("tp",1))
    dp = int(m.get("dp",1))
    pp = int(m.get("pp",1))
    dcp = int(m.get("dcp",1))
    mtp = "true" if m.get("mtp",False) else "false"
    cmd = m.get("command_rest","").replace("\n"," ")
    # 输出多一个字段 dcp
    print("|".join([model,str(tp),str(dp),str(pp),str(dcp),mtp,cmd]))
PY
)

##############################################
# GPU 管理函数
##############################################
get_free_gpus() {
  local MEM_THRESHOLD=900000
  local raw avail out=""
  raw=$(mx-smi --show-memory 2>/dev/null || true)
  avail=$(echo "$raw" | sed 's/：/:/g' | tr -s ' ' \
    | awk -v threshold=$MEM_THRESHOLD '/GPU#/ { gpu_id=substr($1,5) } /vis_vram used/ { used_val=$4; if (used_val+0 < threshold) { free_gpus = (free_gpus=="" ? gpu_id : free_gpus "," gpu_id) } } END{ if(free_gpus) print free_gpus }')

  for g in $(echo "$avail" | tr ',' ' '); do
    [ -z "$g" ] && continue
    if [ ! -f "${LOCK_DIR}/gpu${g}.lock" ]; then
      out="${out}${g},"
    fi
  done
  echo "${out%,}"
}

allocate_gpus() {
  local need=$1
  exec 200>"$ALLOC_LOCK"
  flock -x 200

  local free
  free=$(get_free_gpus)
  IFS=',' read -r -a arr <<< "$free"
  if [ "${#arr[@]}" -lt "$need" ] || [ -z "$free" ]; then
    flock -u 200
    return 1
  fi

  local sel=("${arr[@]:0:$need}")
  for g in "${sel[@]}"; do
    touch "${LOCK_DIR}/gpu${g}.lock"
  done

  flock -u 200
  (IFS=,; echo "${sel[*]}")
  return 0
}

unlock_gpus() {
  local gpus=$1
  for g in $(echo "$gpus" | tr ',' ' '); do
    [ -n "$g" ] && rm -f "${LOCK_DIR}/gpu${g}.lock"
  done
}

##############################################
# 清理函数（退出时安全释放）
##############################################
cleanup() {
  echo "[CLEANUP] 脚本退出，终止后台模型并释放锁..."
  for pid in "${launched_pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      pkill -TERM -P "$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    [ -n "${pid_to_gpus[$pid]:-}" ] && unlock_gpus "${pid_to_gpus[$pid]}"
  done
  rm -f "$ALLOC_LOCK"
}
trap cleanup EXIT INT TERM

##############################################
# 启动模型 online.py （异步 + 日志检测）
##############################################
idx=0
while IFS='|' read -r model tp dp pp dcp mtp command_rest; do
  [ -z "$model" ] && continue
  echo "==================================================="
  echo "[INFO] 模型: $model  tp=$tp dp=$dp pp=$pp dcp=$dcp mtp=$mtp"

  # 校验 dcp 与 tp 的可整除性
  if [ -n "$dcp" ] && [ "$dcp" -gt 1 ]; then
    if [ $((tp % dcp)) -ne 0 ]; then
      echo "[ERROR] 配置不合法：tp=$tp 不能被 dcp=$dcp 整除，跳过模型 $model"
      echo "$(basename "$model")(配置错误: tp% dcp != 0)" >> "$FAILED_FILE"
      continue
    fi
  fi

  need_gpus=$((tp * dp * pp))   # DCP 不改变 world size，保持原计算
  if [ "$need_gpus" -gt "$TOTAL_GPUS" ]; then
    echo "[ERROR] 需要 GPU $need_gpus 超出总量 $TOTAL_GPUS，跳过"
    continue
  fi

  local_gpus=""
  while true; do
    local_gpus=$(allocate_gpus "$need_gpus" || true)
    if [ -n "$local_gpus" ]; then
      echo "[INFO] 原子分配 GPU: $local_gpus"
      break
    fi
    echo "[WAIT] 没有足够空闲 GPU（需 $need_gpus），${WAIT_INTERVAL}s 后重试..."
    sleep $WAIT_INTERVAL
  done

  port=$((PORT_BASE + idx))
  while ss -tuln 2>/dev/null | grep -q ":${port} "; do
    port=$((port + 1))
  done

  log_file="${RUN_LOG_DIR}/$(basename ${model%/})_${port}.log"

  echo "[RUN] 启动模型：$(basename "$model")"
  echo "     GPU 使用：$local_gpus"
  echo "     启动端口：$port"
  echo "     日志路径：$log_file"

  CUDA_VISIBLE_DEVICES=$local_gpus nohup python3 online.py \
    --model_path "$model" \
    --tensor_parallel_size $tp \
    --data_parallel_size $dp \
    --pipeline_parallel_size $pp \
    --decode-context-parallel-size $dcp \
    --port $port \
    --gpus "$local_gpus" \
    --log_file "$log_file" \
    $command_rest >"$log_file" 2>&1 &

  pid=$!
  pid_to_gpus[$pid]="$local_gpus"
  pid_to_log[$pid]="$log_file"
  pid_to_model[$pid]="$model"
  launched_pids+=("$pid")

  echo "✅ 已启动：PID=$pid"
  echo "---------------------------------------------------"

  # 启动独立监控线程
  (
    start_time=$(date +%s)
    max_wait_time=3600  # 最长等待 60 分钟
    success_flag=false
    model_name=$(basename "$model")

    while true; do
      sleep 5
      elapsed=$(( $(date +%s) - start_time ))

      if grep -qiE "Uvicorn running on|Application startup complete|Started server process" "$log_file" 2>/dev/null; then
        echo "✅ [SUCCESS] 模型 ${model_name} 启动成功（PID=$pid）"
        success_flag=true
        break
      fi

      if grep -qiE "Traceback|RuntimeError|Killed|out of memory|OOM|Failed to load|Cannot allocate memory" "$log_file" 2>/dev/null; then
        echo "❌ [FAILED] 模型 ${model_name} 启动失败（检测到报错）"
        break
      fi

      if [ $elapsed -ge $max_wait_time ]; then
        echo "⚠️ [TIMEOUT] 模型 ${model_name} 启动超时（${max_wait_time}s）"
        break
      fi
    done

    if [ "$success_flag" = false ]; then
      echo "$model_name(启动失败)" >> "$FAILED_FILE"
      echo "[CLEAN] 终止 PID=$pid 并释放 GPU：$local_gpus"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      echo "[DELAY] 释放 GPU 中..."
      sleep 60
      unlock_gpus "$local_gpus"
      echo "[SKIP] 已跳过模型 ${model_name}"
      echo "---------------------------------------------------"
    fi
  ) &

  idx=$((idx + 1))
done <<< "$models_lines"

##############################################
# 全局异步监控循环
##############################################
echo "[MONITOR] 开始监控后台模型..."
while :; do
  sleep $CHECK_INTERVAL
  for pid in "${!pid_to_gpus[@]}"; do
    log_file=${pid_to_log[$pid]}
    model=${pid_to_model[$pid]}
    gpus=${pid_to_gpus[$pid]}

    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[MONITOR] 模型 $model (PID=$pid) 已退出，释放 GPU $gpus"
      echo "[SLEEP] 模型 $model 结束,等待30s 后继续"
      sleep 30
      unlock_gpus "$gpus"
      unset pid_to_gpus[$pid]
      continue
    fi

    if grep -q -E "CUDA out of memory|RuntimeError|Killed|Aborted|Segmentation fault" "$log_file"; then
      echo "[FAIL] 模型 $model 运行时失败，释放 GPU $gpus"
      echo "$(basename "$model")(运行时失败)" >> "$FAILED_FILE"
      kill "$pid" 2>/dev/null || true
      unlock_gpus "$gpus"
      unset pid_to_gpus[$pid]
    fi
  done

  [ "${#pid_to_gpus[@]}" -eq 0 ] && break
done

##############################################
# 汇总：检测启动失败 + 精度失败
##############################################
for pid in "${launched_pids[@]}"; do
    log_file=${pid_to_log[$pid]}
    model=${pid_to_model[$pid]}
    if grep -q "❌ \[PRECISION\]" "$log_file"; then
        echo "$(basename "$model")(精度失败)" >> "$FAILED_FILE"
    fi
done

echo "[DONE] 所有模型测试完成"

echo "======================"
echo "测试时间: $(date)"
if [ -s "$FAILED_FILE" ]; then
  echo "[SUMMARY] 以下模型启动或精度失败："
  cat "$FAILED_FILE"
else
  echo "[SUMMARY] 所有模型均启动并精度通过 ✅"
fi

echo "[LOG] 详细日志保存在：$RUN_LOG_DIR"
echo "[LOG] 失败模型文件：$FAILED_FILE"
