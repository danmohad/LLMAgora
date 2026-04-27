#!/usr/bin/env bash
# Matplotlib's text.usetex=True calls latex, dvipng, and ghostscript (gs).
export PATH="/Library/TeX/texbin:${PATH:-}"
# Install (macOS, Homebrew):
#   brew install --cask basictex ghostscript
#   sudo tlmgr update --self
#   sudo tlmgr install dvipng cm-super collection-fontsrecommended type1cm
# (type1cm provides type1cm.sty; without it Matplotlib usetex fails with
#  "File `type1cm.sty' not found".)
# Ensure /Library/TeX/texbin is on PATH (BasicTeX installer usually adds this).

set -e
missing=()
for cmd in latex dvipng gs; do
  if ! command -v "$cmd" &>/dev/null; then
    missing+=("$cmd")
  fi
done
if ((${#missing[@]})); then
  echo "Missing on PATH: ${missing[*]}"
  exit 1
fi
if ! kpsewhich type1cm.sty &>/dev/null; then
  echo "TeX file type1cm.sty not found (package: type1cm)."
  echo "Install with: sudo \"\$(command -v tlmgr)\" install type1cm"
  exit 1
fi
echo "OK: latex, dvipng, gs, and type1cm.sty are available for Matplotlib usetex."
latex --version | head -1
gs --version | head -1
