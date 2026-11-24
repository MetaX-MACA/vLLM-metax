#!/usr/bin/env bash
set -euo pipefail

readonly uv_version=0.11.1
readonly pre_commit_version=4.7.0
readonly shellcheck_version=0.11.0
readonly go_version=1.25.9
readonly release_base=${JENKINS_PRE_COMMIT_RELEASE_BASE:-https://github.com}
readonly python_mirror=https://registry.npmmirror.com/-/binary/python-build-standalone
readonly node_mirror=https://mirrors.huaweicloud.com/nodejs

log() {
    printf '[jenkins-pre-commit] %s\n' "$*"
}

fail() {
    printf '[jenkins-pre-commit] %s\n' "$*" >&2
    exit 2
}

usage() {
    cat <<'USAGE'
Usage:
  jenkins-pre-commit.sh probe
  jenkins-pre-commit.sh prepare <checkout-dir> <state-dir>
  jenkins-pre-commit.sh run <checkout-dir> <state-dir>

Run on a Linux Jenkins agent with Git and flock. To bootstrap missing uv,
ShellCheck, or Go binaries, use x86_64 or aarch64 with curl, tar, gzip, and sha256sum.
Use a writable state directory outside the checkout, such as Jenkins pwd(tmp: true).
Run prepare and run as the same user on the same agent. Prepare can be retried.

The script uses the MetaX proxy and defaults to Tsinghua PyPI, Huawei Cloud
npm and Node.js, and Aliyun Go. When uv needs a managed Python, it tries
npmmirror first. Public hook repositories with domestic mirrors use GitCode
or Gitee. Remaining upstream requests use the proxy.

Agents can set PIP_INDEX_URL, UV_DEFAULT_INDEX, UV_PYTHON_INSTALL_MIRROR,
npm_config_registry, or GOPROXY to select different mirrors. An existing
~/.nodeenvrc takes precedence over the default Node.js mirror.
JENKINS_PRE_COMMIT_RELEASE_BASE can select an internal GitHub release mirror
for the uv and ShellCheck binary bootstraps. PRE_COMMIT_HOME, UV_CACHE_DIR,
and UV_PYTHON_INSTALL_DIR default to subdirectories of the state directory.
USAGE
}

probe() {
    log "Node=${NODE_NAME:-local} User=$(id -un) OS=$(uname -s) Arch=$(uname -m)"
    if [[ -r /etc/os-release ]]; then
        grep '^PRETTY_NAME=' /etc/os-release || true
    fi
    df -h .
    local tool tool_path
    for tool in git curl bash flock tar gzip sha256sum python3 python3.12 uv pre-commit shellcheck node npm go; do
        if tool_path=$(command -v "$tool"); then
            printf '%-12s %s\n' "$tool" "$tool_path"
        else
            printf '%-12s missing\n' "$tool"
        fi
    done
}

download_archive() {
    local url=$1 archive=$2 checksum=$3
    log "Downloading ${url##*/}"
    curl --fail --silent --show-error --location --retry 2 --retry-all-errors \
        --connect-timeout 15 --max-time 180 --output "$archive" "$url"
    printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check --status \
        || fail "SHA-256 verification failed for ${url##*/}."
}

install_release_binary() {
    local release_path=$1 archive_member=$2 checksum=$3 target=$4
    local archive="$state_dir/${target##*/}.tar.gz"
    download_archive "${release_base%/}/$release_path" "$archive" "$checksum"
    tar -xOzf "$archive" -- "$archive_member" > "$target.new"
    chmod 755 "$target.new"
    mv -f -- "$target.new" "$target"
    rm -f -- "$archive"
}

prepare_uv() {
    if command -v uv >/dev/null; then
        return
    fi
    if [[ -z ${JENKINS_PRE_COMMIT_RELEASE_BASE:-} ]] &&
        command -v python3 >/dev/null && python3 -m pip --version >/dev/null 2>&1 &&
        python3 -m pip install --target "$state_dir/uv" --upgrade --no-deps --only-binary=:all: "uv==$uv_version"; then
        return
    fi

    local arch checksum asset
    arch=$(uname -m)
    case "$arch" in
        x86_64) checksum=7c0c8069053e6e99e5911ff32b916be571f3419cd8e11bd28fb7da2c7dcaa553 ;;
        aarch64) checksum=1340e62da1ee3c1109764340e1247e8a1a232c30dde4a0f0548976dcaa90f06d ;;
        *) fail "Install uv on the Agent for architecture $arch." ;;
    esac
    asset="uv-$arch-unknown-linux-gnu"
    install_release_binary "astral-sh/uv/releases/download/$uv_version/$asset.tar.gz" \
        "$asset/uv" "$checksum" "$state_dir/bin/uv"
}

prepare_shellcheck() {
    if command -v shellcheck >/dev/null; then
        return
    fi
    if [[ -z ${JENKINS_PRE_COMMIT_RELEASE_BASE:-} ]] &&
        uv pip install --python "$state_dir/venv/bin/python3.12" --no-deps --only-binary=:all: 'shellcheck-py==0.11.0.1'; then
        return
    fi

    local arch checksum asset="shellcheck-v$shellcheck_version"
    arch=$(uname -m)
    case "$arch" in
        x86_64) checksum=b7af85e41cc99489dcc21d66c6d5f3685138f06d34651e6d34b42ec6d54fe6f6 ;;
        aarch64) checksum=68a8133197a50beb8803f8d42f9908d1af1c5540d4bb05fdfca8c1fa47decefc ;;
        *) fail "Install ShellCheck on the Agent for architecture $arch." ;;
    esac
    install_release_binary "koalaman/shellcheck/releases/download/v$shellcheck_version/$asset.linux.$arch.tar.gz" \
        "$asset/shellcheck" "$checksum" "$state_dir/bin/shellcheck"
}

prepare_python() {
    local venv="$state_dir/venv"
    if [[ -x $venv/bin/python3.12 ]]; then
        return
    fi
    if [[ -n ${UV_PYTHON_INSTALL_MIRROR:-} ]]; then
        uv venv --no-project --clear --python 3.12 "$venv"
    elif ! UV_PYTHON_INSTALL_MIRROR="$python_mirror" uv venv --no-project --clear --python 3.12 "$venv"; then
        log 'Python mirror unavailable, retrying via GitHub through the proxy'
        UV_PYTHON_INSTALL_MIRROR=https://github.com/astral-sh/python-build-standalone/releases/download \
            uv venv --no-project --clear --python 3.12 "$venv"
    fi
}

prepare_go() (
    if command -v go >/dev/null; then
        return
    fi

    local arch checksum asset staging
    case "$(uname -m)" in
        x86_64)
            arch=amd64
            checksum=00859d7bd6defe8bf84d9db9e57b9a4467b2887c18cd93ae7460e713db774bc1
            ;;
        aarch64)
            arch=arm64
            checksum=ec342e7389b7f489564ed5463c63b16fc8040023abc7861256677165a8c0e2b
            ;;
        *) fail 'Install Go on the Agent for this architecture.' ;;
    esac
    asset="go$go_version.linux-$arch.tar.gz"
    staging=$(mktemp -d "$state_dir/.go.XXXXXXXX")
    trap 'rm -rf -- "$staging"' EXIT
    download_archive "https://mirrors.aliyun.com/golang/$asset" "$staging/$asset" "$checksum"
    tar -xzf "$staging/$asset" -C "$staging"
    "$staging/go/bin/go" version
    mv -T -- "$staging/go" "$state_dir/go"
)

add_git_mirror() {
    local index=${GIT_CONFIG_COUNT:-0}
    export "GIT_CONFIG_KEY_$index=url.$2.insteadOf" "GIT_CONFIG_VALUE_$index=https://github.com/$1"
    export GIT_CONFIG_COUNT=$((index + 1))
}

run_hooks() (
    local nodeenv_config='' go_links='' hook_path=$PATH
    local nodeenv_lock_dir="$HOME/.cache/jenkins-pre-commit"
    trap 'if [[ -n $nodeenv_config ]]; then
        if [[ $(readlink -- "$HOME/.nodeenvrc") == "$nodeenv_config" ]]; then
            rm -f -- "$HOME/.nodeenvrc"
        fi
        rm -f -- "$nodeenv_config"
    fi
    if [[ -n $go_links ]]; then
        rm -rf -- "$go_links"
    fi' EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM

    # Serialize use of the temporary nodeenv configuration across jobs for this user.
    mkdir -p -- "$nodeenv_lock_dir"
    exec 8>>"$nodeenv_lock_dir/nodeenv.lock"
    flock 8
    if [[ ! -e $HOME/.nodeenvrc && ! -L $HOME/.nodeenvrc ]]; then
        nodeenv_config=$(mktemp "$state_dir/.nodeenvrc.XXXXXXXX")
        printf '[nodeenv]\nmirror = %s\n' "$node_mirror" > "$nodeenv_config"
        ln -sT -- "$nodeenv_config" "$HOME/.nodeenvrc"
    fi

    # pre-commit recognizes Go as a system tool only when its path is outside HOME.
    if ! python3.12 -c 'from pre_commit.lang_base import exe_exists; raise SystemExit(not exe_exists("go"))'; then
        go_links=$(mktemp -d /var/tmp/jenkins-pre-commit-go.XXXXXXXX)
        ln -s -- "$(command -v go)" "$go_links/go"
        hook_path="$go_links:$hook_path"
    fi
    PATH="$hook_path" python3.12 -m pre_commit "$@" 8<&-
)

case "${1:-}" in
    probe)
        [[ $# == 1 ]] || fail 'Usage: jenkins-pre-commit.sh probe'
        probe
        exit 0
        ;;
    prepare|run)
        [[ $# == 3 ]] || { usage >&2; exit 2; }
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

mode=$1
[[ $(uname -s) == Linux ]] || fail 'A Linux Jenkins Agent is required.'
for tool in git realpath flock; do
    command -v "$tool" >/dev/null || fail "Required Agent tool missing: $tool"
done
repo_dir=$(git -C "$2" rev-parse --show-toplevel) || fail "Git checkout unavailable: $2"
[[ -f $repo_dir/.pre-commit-config.yaml ]] || fail "No .pre-commit-config.yaml in $repo_dir"
state_dir=$(realpath -m -- "$3")
case "$state_dir/" in
    "$repo_dir/"*) fail 'Choose a state directory outside the Git checkout.' ;;
esac
mkdir -p -- "$state_dir/bin"
exec 9>"$state_dir/.lock"
flock 9

export PRE_COMMIT_HOME=${PRE_COMMIT_HOME:-"$state_dir/hooks"}
export UV_CACHE_DIR=${UV_CACHE_DIR:-"$state_dir/uv-cache"}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-"$state_dir/python"}
export GIT_TERMINAL_PROMPT=0 PIP_NO_INPUT=1
export https_proxy="http://proxy.sha-dc01.metax-tech.com:1080"
export http_proxy="${http_proxy:-$https_proxy}"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export no_proxy="${no_proxy:-${NO_PROXY:-localhost,127.0.0.1,::1,.metax-tech.com,.mxcr.io,.metax-internal.com,.infra.dev,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16}}"
export NO_PROXY="$no_proxy"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-$PIP_INDEX_URL}"
export npm_config_registry="${npm_config_registry:-https://mirrors.huaweicloud.com/repository/npm/}"
export GOPROXY="${GOPROXY:-https://mirrors.aliyun.com/goproxy/|https://proxy.golang.org|direct}"

add_git_mirror astral-sh/ruff-pre-commit https://gitcode.com/gh_mirrors/ru/ruff-pre-commit
add_git_mirror crate-ci/typos https://gitcode.com/gh_mirrors/ty/typos
add_git_mirror pre-commit/mirrors-clang-format https://gitee.com/mirrors_pre-commit/mirrors-clang-format
add_git_mirror DavidAnson/markdownlint-cli2 https://gitcode.com/gh_mirrors/ma/markdownlint-cli2
add_git_mirror rhysd/actionlint https://gitcode.com/gh_mirrors/ac/actionlint

unset PYTHONHOME PYTHONPATH SKIP
export PATH="$state_dir/venv/bin:$state_dir/bin:$state_dir/uv/bin:$state_dir/go/bin:$PATH"

if [[ $mode == prepare ]]; then
    log "Preparing tools on ${NODE_NAME:-local}"
    # Bootstrap from the state directory so the patchset cannot change the uv setup.
    cd -- "$state_dir"
    prepare_uv
    uv --version
    prepare_python
    uv pip install --python "$state_dir/venv/bin/python3.12" "pre-commit==$pre_commit_version"
    prepare_shellcheck
fi

[[ -x $state_dir/venv/bin/python3.12 ]] || fail 'Run prepare on this Agent first.'
command -v shellcheck >/dev/null || fail 'ShellCheck missing. Run prepare on this Agent first.'
python3.12 --version
python3.12 -m pre_commit --version
shellcheck --version
prepare_go
cd -- "$repo_dir"
log "Checkout $(git rev-parse HEAD)"

if [[ $mode == prepare ]]; then
    log "Installing hook environments from .pre-commit-config.yaml"
    python3.12 -m pre_commit validate-config .pre-commit-config.yaml
    run_hooks install-hooks
else
    log 'Running manual-stage hooks on all tracked files'
    run_hooks run --all-files --hook-stage manual --show-diff-on-failure
fi
