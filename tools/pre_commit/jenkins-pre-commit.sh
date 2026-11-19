#!/usr/bin/env bash
set -euo pipefail

readonly uv_version=0.11.1
readonly pre_commit_version=4.7.0
readonly shellcheck_version=0.11.0
readonly release_base=${JENKINS_PRE_COMMIT_RELEASE_BASE:-https://github.com}

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

Run on a Linux Jenkins agent with Git and flock. To bootstrap missing uv and
ShellCheck binaries, use x86_64 or aarch64 with curl, tar, gzip, and sha256sum.
Use a writable state directory outside the checkout, such as Jenkins pwd(tmp: true).
Run prepare and run as the same user on the same agent. Prepare can be retried.

JENKINS_PRE_COMMIT_RELEASE_BASE can point to an internal mirror of GitHub
release assets. Standard Git, pip, and uv mirror settings still apply to hook
environments. PRE_COMMIT_HOME, UV_CACHE_DIR, and UV_PYTHON_INSTALL_DIR default
to subdirectories of the state directory and can be set by the agent.
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

install_release_binary() {
    local release_path=$1 archive_member=$2 checksum=$3 target=$4
    local archive="$state_dir/${target##*/}.tar.gz"
    log "Downloading ${release_path##*/}"
    curl --fail --silent --show-error --location --retry 2 --retry-all-errors \
        --connect-timeout 15 --max-time 180 \
        --output "$archive" "${release_base%/}/$release_path"
    printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check --status \
        || fail "SHA-256 verification failed for ${release_path##*/}."
    tar -xOzf "$archive" -- "$archive_member" > "$target.new"
    chmod 755 "$target.new"
    mv -f -- "$target.new" "$target"
    rm -f -- "$archive"
}

prepare_tools() {
    local arch uv_sha shellcheck_sha uv_asset shellcheck_asset
    arch=$(uname -m)
    case "$arch" in
        x86_64)
            uv_sha=7c0c8069053e6e99e5911ff32b916be571f3419cd8e11bd28fb7da2c7dcaa553
            shellcheck_sha=b7af85e41cc99489dcc21d66c6d5f3685138f06d34651e6d34b42ec6d54fe6f6
            ;;
        aarch64)
            uv_sha=1340e62da1ee3c1109764340e1247e8a1a232c30dde4a0f0548976dcaa90f06d
            shellcheck_sha=68a8133197a50beb8803f8d42f9908d1af1c5540d4bb05fdfca8c1fa47decefc
            ;;
    esac
    if ! command -v uv >/dev/null; then
        [[ -n ${uv_sha:-} ]] || fail "Install uv on the Agent for architecture $arch."
        uv_asset="uv-$arch-unknown-linux-gnu"
        install_release_binary "astral-sh/uv/releases/download/$uv_version/$uv_asset.tar.gz" \
            "$uv_asset/uv" "$uv_sha" "$state_dir/bin/uv"
    fi
    if ! command -v shellcheck >/dev/null; then
        [[ -n ${shellcheck_sha:-} ]] || fail "Install ShellCheck on the Agent for architecture $arch."
        shellcheck_asset="shellcheck-v$shellcheck_version"
        install_release_binary "koalaman/shellcheck/releases/download/v$shellcheck_version/$shellcheck_asset.linux.$arch.tar.gz" \
            "$shellcheck_asset/shellcheck" "$shellcheck_sha" "$state_dir/bin/shellcheck"
    fi
}

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
unset PYTHONHOME PYTHONPATH SKIP
export PATH="$state_dir/venv/bin:$state_dir/bin:$PATH"

if [[ $mode == prepare ]]; then
    log "Preparing tools on ${NODE_NAME:-local}"
    prepare_tools
    uv --version
    # Bootstrap from the state directory so the patchset cannot change the uv setup.
    cd -- "$state_dir"
    if [[ ! -x $state_dir/venv/bin/python3.12 ]]; then
        uv venv --no-project --clear --python 3.12 "$state_dir/venv"
    fi
    uv pip install --python "$state_dir/venv/bin/python3.12" "pre-commit==$pre_commit_version"
fi

[[ -x $state_dir/venv/bin/python3.12 ]] || fail 'Run prepare on this Agent first.'
command -v shellcheck >/dev/null || fail 'ShellCheck missing. Run prepare on this Agent first.'
python3.12 --version
python3.12 -m pre_commit --version
shellcheck --version
cd -- "$repo_dir"
log "Checkout $(git rev-parse HEAD)"

if [[ $mode == prepare ]]; then
    log "Installing hook environments from .pre-commit-config.yaml"
    python3.12 -m pre_commit validate-config .pre-commit-config.yaml
    python3.12 -m pre_commit install-hooks
else
    log 'Running manual-stage hooks on all tracked files'
    python3.12 -m pre_commit run --all-files --hook-stage manual --show-diff-on-failure
fi
