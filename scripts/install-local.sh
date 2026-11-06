#!/usr/bin/env bash
# Install the omnifrog you use every day, out of cargo's target/.
#
# The menu entry, and anything else told where omnifrog is, run the installed
# copy, so cleaning or rebuilding a checkout never breaks them. A build under
# target/ (cargo build, cargo run) is a dev copy.
#
#   scripts/install-local.sh               build main and install it
#   scripts/install-local.sh --any-branch  install the current branch anyway
#
# Installs into $PREFIX (default ~/.local):
#   bin/omnifrog
#   bin/omnifrog-gui
#   share/applications/omnifrog.desktop
#   share/icons/hicolor/scalable/apps/omnifrog.svg
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"
# A relative PREFIX is read from where the script was started; the menu entry
# runs from anywhere, so it has to name an absolute path.
case "$PREFIX" in
  /*) ;;
  *) PREFIX="$PWD/$PREFIX" ;;
esac

any_branch=0
case "${1:-}" in
  "") ;;
  --any-branch) any_branch=1 ;;
  -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
esac

cd "$REPO"
branch="$(git symbolic-ref --short -q HEAD || echo '(detached HEAD)')"
if [ "$any_branch" = 0 ]; then
  if [ "$branch" != main ]; then
    echo "on $branch, not main: the installed copy is the one you rely on." >&2
    echo "Switch to main, or pass --any-branch to install this branch anyway." >&2
    exit 1
  fi
  if [ -n "$(git status --porcelain)" ]; then
    echo "the checkout has uncommitted changes; commit or stash them first," >&2
    echo "or pass --any-branch to install them anyway." >&2
    exit 1
  fi
fi
version="$(git describe --always --dirty)"

# The programs are where Cargo says it put them: a target-dir or a build target
# in a Cargo configuration moves them out of target/release.
messages="$(cargo build --release --locked -p omnifrog-cli -p omnifrog-gui \
  --message-format=json-render-diagnostics)"
built() { # name → the executable Cargo reported for it in this build
  printf '%s\n' "$messages" | grep '"reason":"compiler-artifact"' |
    sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' | grep "/$1\$" | tail -n 1 || true
}
cli_built="$(built omnifrog)"
gui_built="$(built omnifrog-gui)"
for path in "$cli_built" "$gui_built"; do
  if [ -z "$path" ] || [ ! -f "$path" ]; then
    echo "cargo did not report both omnifrog and omnifrog-gui; nothing installed" >&2
    exit 1
  fi
done

install -d "$PREFIX/bin"
for pair in "omnifrog:$cli_built" "omnifrog-gui:$gui_built"; do
  name="${pair%%:*}"
  # Copied beside the old one and renamed over it: an open window keeps its
  # file, and its worker (started through /proc/self/exe) is still its own.
  install -m 755 "${pair#*:}" "$PREFIX/bin/$name.new"
  mv -f "$PREFIX/bin/$name.new" "$PREFIX/bin/$name"
done
gui="$PREFIX/bin/omnifrog-gui"

install -D -m 644 packaging/linux/omnifrog.svg \
  "$PREFIX/share/icons/hicolor/scalable/apps/omnifrog.svg"

# The packaged entry runs omnifrog-gui from PATH; this one names the copy just
# installed, which PATH may not find first.
template=packaging/linux/omnifrog.desktop
grep -q '^Exec=omnifrog-gui ' "$template" || {
  echo "$template has no 'Exec=omnifrog-gui …' line to point at $gui" >&2
  exit 1
}
case "$gui" in
  *[\"\`\$\\%]*)
    echo "a desktop entry cannot name $gui as it is; choose another PREFIX" >&2
    exit 1 ;;
esac
entry="$PREFIX/share/applications/omnifrog.desktop"
install -d "$(dirname "$entry")"
# Quoted for a PREFIX with spaces; & and | escaped for sed.
exec_path="$(printf '"%s"' "$gui" | sed 's/[&|]/\\&/g')"
sed "s|^Exec=omnifrog-gui |Exec=$exec_path |" "$template" > "$entry.new"
mv -f "$entry.new" "$entry"

command -v update-desktop-database >/dev/null &&
  update-desktop-database -q "$PREFIX/share/applications" || true
command -v kbuildsycoca6 >/dev/null && kbuildsycoca6 >/dev/null 2>&1 || true

echo "installed omnifrog $version ($branch) as $PREFIX/bin/omnifrog and $gui," \
  "built in $(dirname "$gui_built")"
for name in omnifrog omnifrog-gui; do
  found="$(command -v "$name" || true)"
  if [ "$found" != "$PREFIX/bin/$name" ]; then
    echo "note: $name on PATH is ${found:-nothing}, not $PREFIX/bin/$name;" \
      "name the installed copy by its full path where it matters." >&2
  fi
done
