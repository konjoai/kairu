#!/usr/bin/env bash
# Konjo Quality Framework — Hook Installer
#
# Usage (from any repo root):
#   bash .konjo/scripts/install-hooks.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
GRN='\033[0;32m'; YEL='\033[0;33m'; RED='\033[0;31m'; RST='\033[0m'; BOLD='\033[1m'

ok()   { echo -e "${GRN}  ✓${RST} $1"; }
warn() { echo -e "${YEL}  ⚠${RST} $1"; }
err()  { echo -e "${RED}  ✗${RST} $1"; }

echo -e "${BOLD}Konjo Quality Framework — Install${RST}"
echo ""

HOOK_SRC="$REPO_ROOT/.konjo/hooks/pre-commit"
HOOK_DST="$REPO_ROOT/.git/hooks/pre-commit"

if [[ ! -f "$HOOK_SRC" ]]; then
    err ".konjo/hooks/pre-commit not found — copy the .konjo/ directory first"
    exit 1
fi

chmod +x "$HOOK_SRC"

if [[ -L "$HOOK_DST" ]]; then
    rm "$HOOK_DST"
fi

ln -sf "../../.konjo/hooks/pre-commit" "$HOOK_DST"
ok "Installed .git/hooks/pre-commit → .konjo/hooks/pre-commit"

HAS_RUST=false; HAS_PYTHON=false; HAS_MOJO=false
[[ -f "$REPO_ROOT/Cargo.toml" ]] && HAS_RUST=true
{ [[ -f "$REPO_ROOT/pyproject.toml" ]] || [[ -f "$REPO_ROOT/requirements.txt" ]]; } && HAS_PYTHON=true
[[ -f "$REPO_ROOT/pixi.toml" ]] && HAS_MOJO=true

echo ""
echo -e "${BOLD}Repo type:${RST}"
$HAS_RUST    && ok "Rust" || true
$HAS_PYTHON  && ok "Python" || true
$HAS_MOJO    && ok "Mojo" || true

echo ""
echo -e "${BOLD}Tool availability:${RST}"
ALL_PRESENT=true

check_tool() {
    local cmd="$1" install_hint="$2"
    if command -v "$cmd" &>/dev/null; then
        ok "$cmd"
    else
        err "$cmd not found — $install_hint"
        ALL_PRESENT=false
    fi
}

check_tool "python3" "install Python 3.10+"
check_tool "git" "install git"

if $HAS_RUST; then
    check_tool "cargo" "install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi

if $HAS_PYTHON; then
    check_tool "ruff" "pip install ruff"
    check_tool "mypy" "pip install mypy"
    check_tool "vulture" "pip install vulture"
    check_tool "radon" "pip install radon"
fi

if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    ok "ANTHROPIC_API_KEY — set ✓"
else
    warn "ANTHROPIC_API_KEY not set — Wall 3 adversarial review requires it"
    warn "  export ANTHROPIC_API_KEY=your_key"
fi

echo ""
if $ALL_PRESENT; then
    echo -e "${GRN}${BOLD}All required tools present. Framework installed.${RST}"
else
    echo -e "${YEL}${BOLD}Some tools missing — install them before the full gate runs.${RST}"
fi

echo ""
echo "Next steps:"
echo "  1. Add ANTHROPIC_API_KEY to GitHub Actions secrets"
echo "  2. Verify .github/workflows/konjo-gate.yml is present"
echo "  3. Run: git commit --allow-empty -m 'test: verify konjo hooks'"
echo ""
echo "Docs: KONJO_QUALITY_FRAMEWORK.md"
