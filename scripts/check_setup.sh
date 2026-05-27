#!/usr/bin/env bash
# ------------------------------------------------------------------
# check_setup.sh – verify prerequisites before initialising the env
# ------------------------------------------------------------------
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No colour

ok=0
warn=0
fail=0

pass()  { ok=$((ok + 1));   printf "${GREEN}[OK]${NC}   %s\n" "$1"; }
skip()  { warn=$((warn + 1)); printf "${YELLOW}[WARN]${NC} %s\n" "$1"; }
die()   { fail=$((fail + 1)); printf "${RED}[FAIL]${NC} %s\n" "$1"; }

# ---- 1. git -----------------------------------------------------------
if command -v git &>/dev/null; then
    pass "git is installed ($(git --version | head -1))"
else
    die "git is not installed"
fi

# ---- 2. uv ------------------------------------------------------------
MIN_UV="0.4.0"

version_gte() {
    # Returns 0 (true) if $1 >= $2 using sort -V
    [ "$(printf '%s\n%s' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

if command -v uv &>/dev/null; then
    uv_version=$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if [ -z "$uv_version" ]; then
        skip "uv is installed but could not parse version"
    elif version_gte "$uv_version" "$MIN_UV"; then
        pass "uv $uv_version installed (>= $MIN_UV required)"
    else
        die "uv $uv_version is too old (>= $MIN_UV required)"
    fi
else
    die "uv is not installed – see https://docs.astral.sh/uv/getting-started/installation/"
fi

# ---- 3. Python version ------------------------------------------------
REQUIRED_MINOR_MIN=11
REQUIRED_MINOR_MAX=12

check_python() {
    local py="$1"
    if command -v "$py" &>/dev/null; then
        local ver
        ver=$("$py" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        local major minor
        major=${ver%%.*}
        minor=${ver#*.}
        if [ "$major" = "3" ] && [ "$minor" -ge "$REQUIRED_MINOR_MIN" ] && [ "$minor" -le "$REQUIRED_MINOR_MAX" ]; then
            pass "Python $ver found via '$py' (3.${REQUIRED_MINOR_MIN}–3.${REQUIRED_MINOR_MAX} required)"
            return 0
        fi
    fi
    return 1
}

python_found=false
for candidate in python3.12 python3.11 python3 python; do
    if check_python "$candidate"; then
        python_found=true
        break
    fi
done

if [ "$python_found" = false ]; then
    # uv can install Python itself; warn but don't fail hard
    skip "No Python 3.${REQUIRED_MINOR_MIN}–3.${REQUIRED_MINOR_MAX} found locally (uv can install one for you)"
fi

# ---- 4. pyproject.toml presence ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
    pass "pyproject.toml found"
else
    skip "pyproject.toml not found – run 'uv init' or create one manually"
fi

# ---- 5. macOS-specific: Xcode CLI tools (needed for native builds) -----
if [ "$(uname -s)" = "Darwin" ]; then
    if xcode-select -p &>/dev/null; then
        pass "Xcode Command Line Tools installed"
    else
        skip "Xcode Command Line Tools not found – run: xcode-select --install"
    fi
fi

# ---- Summary -----------------------------------------------------------
echo ""
echo "──────────────────────────────────────"
printf "Results: ${GREEN}%d passed${NC}, ${YELLOW}%d warnings${NC}, ${RED}%d failed${NC}\n" "$ok" "$warn" "$fail"
echo "──────────────────────────────────────"

if [ "$fail" -gt 0 ]; then
    echo ""
    printf "${RED}Fix the failures above before proceeding.${NC}\n"
    exit 1
fi

if [ "$warn" -gt 0 ]; then
    echo ""
    printf "${YELLOW}Warnings above are non-blocking but worth reviewing.${NC}\n"
fi

exit 0
