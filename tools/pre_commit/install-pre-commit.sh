# Run with sh from the repository root.
set -e

fail() {
  printf '%s\n' "$*" >&2
  exit 1
}

private_directory() {
  (umask 077; mkdir -p -- "$1")
  if [ -L "$1" ] || [ "$(stat -c '%u:%a' "$1")" != "$(id -u):700" ]; then
    fail "Expected a private, user-owned directory at $1."
  fi
}

# Pass the URL, destination, SHA-256, and timeout in seconds.
download_archive() {
  [ -n "$3" ] || fail "Install the missing tool for $(uname -sm) and add it to PATH."
  curl -fsSL --retry 2 --connect-timeout 15 --max-time "$4" -o "$2" "$1"
  printf '%s  %s\n' "$3" "$2" | sha256sum -c -
}

# Mirrors and per-user caches.
[ -n "${https_proxy:-}" ] || fail 'Set https_proxy before running this file.'
export PATH="$PATH:$HOME/.local/bin:/opt/conda/bin:/usr/local/go/bin:/usr/lib/go-1.24/bin:/usr/local/bin"
export https_proxy
export http_proxy="${http_proxy:-$https_proxy}"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export no_proxy="localhost,127.0.0.1,::1,.metax-tech.com,.mxcr.io,.metax-internal.com,.infra.dev,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
pypi_mirror="https://pypi.tuna.tsinghua.edu.cn/simple"
npm_mirror="https://mirrors.huaweicloud.com/repository/npm/"
go_proxy="https://mirrors.aliyun.com/goproxy/|https://proxy.golang.org|direct"
export PIP_INDEX_URL="${PIP_INDEX_URL:-$pypi_mirror}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-$PIP_INDEX_URL}"
export npm_config_registry="${npm_config_registry:-$npm_mirror}"
export GOPROXY="${GOPROXY:-$go_proxy}"

cache_root="${XDG_CACHE_HOME:-$HOME/.cache}"
tool_cache="$cache_root/vllm-metax-pre-commit"
(umask 077; mkdir -p -- "$tool_cache")
exec 9>"$tool_cache/install.lock"
flock 9
export PATH="$PATH:$tool_cache/bin:$tool_cache/shellcheck/bin"

if [ -z "${PRE_COMMIT_HOME:-}" ] && [ "$(stat -f -c %T "$cache_root")" = nfs ]; then
  PRE_COMMIT_HOME="/tmp/vllm-metax-pre-commit-$(id -u)"
  private_directory "$PRE_COMMIT_HOME"
  export PRE_COMMIT_HOME
fi

setup_dir=$(mktemp -d)
go_stage=
nodeenv_config="$setup_dir/nodeenvrc"
cleanup() {
  if [ -L "$HOME/.nodeenvrc" ] && [ "$(readlink "$HOME/.nodeenvrc")" = "$nodeenv_config" ]; then
    rm -- "$HOME/.nodeenvrc"
  fi
  rm -rf -- "$setup_dir"
  if [ -n "$go_stage" ]; then
    rm -rf -- "$go_stage"
  fi
}
trap cleanup 0
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# Upstream release checksums. Used only when a tool needs a binary download.
arch=$(uname -m)
case "$(uname -s):$arch" in
  Linux:x86_64)
    go_arch=amd64
    uv_sha=4e949471a95b37088a1ff1a585f69abed4d3cd3f921f50709a46b6ba62986d38
    go_sha=00859d7bd6defe8bf84d9db9e57b9a4467b2887c18cd93ae7460e713db774bc1
    shellcheck_sha=8c3be12b05d5c177a04c29e3c78ce89ac86f1595681cab149b65b97c4e227198
    ;;
  Linux:aarch64)
    go_arch=arm64
    uv_sha=bd04ffce77ee8d77f39823c13606183581847c2f5dcd704f2ea0f15e376b1a27
    go_sha=ec342e7389b7f489564ed5463c63b16cf8040023dabc7861256677165a8c0e2b
    shellcheck_sha=12b331c1d2db6b9eb13cfca64306b1b157a86eb69db83023e261eaa7e7c14588
    ;;
  *)
    go_arch=''
    uv_sha=''
    go_sha=''
    shellcheck_sha=''
    ;;
esac

# Install pre-commit into an isolated Python environment.
uv_bin=$(command -v uv || printf '%s' "$tool_cache/uv/bin/uv")
if [ ! -x "$uv_bin" ]; then
  bootstrap_python=$(command -v python || command -v python3 || true)
  if [ -n "$bootstrap_python" ] &&
     "$bootstrap_python" -c 'import pip, sys; sys.exit(sys.version_info < (3, 8))' >/dev/null 2>&1; then
    "$bootstrap_python" -m pip install --target "$tool_cache/uv" --only-binary=:all: 'uv==0.11.1'
  else
    release="uv-$arch-unknown-linux-musl"
    download_archive "https://github.com/astral-sh/uv/releases/download/0.11.1/$release.tar.gz" \
      "$setup_dir/uv.tar.gz" "$uv_sha" 600
    tar -xzOf "$setup_dir/uv.tar.gz" "$release/uv" > "$setup_dir/uv"
    mkdir -p -- "$(dirname -- "$uv_bin")"
    install -m 0755 "$setup_dir/uv" "$uv_bin"
  fi
fi

venv="$tool_cache/venv"
python_bin="$venv/bin/python"
if [ ! -x "$python_bin" ]; then
  "$uv_bin" venv --python '>=3.11,<3.13' "$venv"
fi
"$uv_bin" pip install --python "$python_bin" 'pre-commit==4.7.0'
export PATH="$venv/bin:$PATH"

# Use system Go when available. Otherwise keep the toolchain in the user cache.
if ! command -v go >/dev/null 2>&1; then
  release="go1.25.9.linux-$go_arch"
  go_dir="$tool_cache/$release"
  if [ ! -x "$go_dir/bin/go" ]; then
    download_archive "https://mirrors.aliyun.com/golang/$release.tar.gz" \
      "$setup_dir/go.tar.gz" "$go_sha" 300
    go_stage=$(mktemp -d "$tool_cache/.go.XXXXXXXX")
    tar -xzf "$setup_dir/go.tar.gz" -C "$go_stage"
    "$go_stage/go/bin/go" version
    mv -T -- "$go_stage/go" "$go_dir"
  fi
  export PATH="$go_dir/bin:$PATH"
fi

# pre-commit ignores Go paths under HOME when it selects the system toolchain.
recognizes_go() {
  "$python_bin" -c 'from pre_commit.lang_base import exe_exists; raise SystemExit(not exe_exists("go"))'
}
if ! recognizes_go; then
  goroot=$(GOTOOLCHAIN=local go env GOROOT)
  go_binary=$(readlink -f -- "$goroot/bin/go")
  [ -x "$go_binary" ] || fail "Go is unavailable at $go_binary."
  go_key=$(printf '%s' "$go_binary" | sha256sum | cut -d ' ' -f 1)
  go_links="/var/tmp/vllm-metax-pre-commit-tools-$(id -u)"
  private_directory "$go_links"
  go_link_dir="$go_links/$go_key"
  mkdir -p -- "$go_link_dir"
  go_link="$go_link_dir/go"
  if ! ln -sT -- "$go_binary" "$go_link" 2>/dev/null; then
    [ -L "$go_link" ] && [ "$(readlink "$go_link")" = "$go_binary" ] ||
      fail "Unexpected contents at $go_link."
  fi
  export PATH="$go_link_dir:$PATH"
  recognizes_go || fail "pre-commit could not recognize Go at $go_link."
fi

# Prefer the PyPI wheel for ShellCheck and fall back to its upstream binary.
if ! command -v shellcheck >/dev/null 2>&1 &&
   ! "$uv_bin" pip install --target "$tool_cache/shellcheck" --no-deps --only-binary=:all: 'shellcheck-py==0.11.0.1'; then
  release="shellcheck-v0.11.0.linux.$arch"
  download_archive "https://github.com/koalaman/shellcheck/releases/download/v0.11.0/$release.tar.xz" \
    "$setup_dir/shellcheck.tar.xz" "$shellcheck_sha" 180
  tar -xJOf "$setup_dir/shellcheck.tar.xz" --wildcards '*/shellcheck' > "$setup_dir/shellcheck"
  mkdir -p -- "$tool_cache/bin"
  install -m 0755 "$setup_dir/shellcheck" "$tool_cache/bin/shellcheck"
fi

# Build the hook environments and register both Git hooks.
if [ ! -e "$HOME/.nodeenvrc" ] && [ ! -L "$HOME/.nodeenvrc" ]; then
  printf '[nodeenv]\nmirror = https://mirrors.huaweicloud.com/nodejs\n' > "$nodeenv_config"
  ln -sT -- "$nodeenv_config" "$HOME/.nodeenvrc"
fi
"$python_bin" -m pre_commit install-hooks
"$python_bin" -m pre_commit install --hook-type pre-commit --hook-type commit-msg

# Keep tool paths and public mirror defaults available in new terminals.
"$python_bin" - "$(git rev-parse --git-path hooks)" "$pypi_mirror" "$npm_mirror" "$go_proxy" <<'PY'
import os
import shlex
import shutil
import sys
from pathlib import Path
from stat import S_IMODE
from tempfile import TemporaryDirectory

hook_dir, pypi, npm, go_proxy = sys.argv[1:]
tools = (sys.executable, shutil.which("go"), shutil.which("shellcheck"))
bin_dirs = dict.fromkeys(str(Path(tool).parent) for tool in tools if tool)
setup = f'export PATH={shlex.quote(os.pathsep.join(bin_dirs))}:"$PATH"\n'
defaults = {
    "PIP_INDEX_URL": pypi,
    "UV_DEFAULT_INDEX": pypi,
    "npm_config_registry": npm,
    "GOPROXY": go_proxy,
}
if cache := os.environ.get("PRE_COMMIT_HOME"):
    defaults["PRE_COMMIT_HOME"] = str(Path(cache).resolve())
for name, value in defaults.items():
    setup += f'if [ -z "${{{name}:-}}" ]; then\n    export {name}={shlex.quote(value)}\nfi\n'

marker = "# end templated\n"
with TemporaryDirectory(dir=hook_dir) as directory:
    for name in ("pre-commit", "commit-msg"):
        hook = Path(hook_dir) / name
        content = hook.read_text()
        if marker not in content:
            raise SystemExit(f"Could not find the pre-commit header in {hook}")
        updated = Path(directory) / name
        updated.write_text(content.replace(marker, setup + marker, 1))
        updated.chmod(S_IMODE(hook.stat().st_mode))
        updated.replace(hook)
PY

# Check staged files.
"$python_bin" -m pre_commit run
