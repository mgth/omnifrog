#!/usr/bin/env bash
# The Linux AppImage: the command line and the window in one file.
#
#   packaging/linux/appimage.sh BIN_DIR VERSION OUT.AppImage
#
# BIN_DIR holds the release builds of omnifrog and omnifrog-gui. The
# appimagetool AppImage and the runtime it embeds are given by APPIMAGETOOL
# and APPIMAGE_RUNTIME, so the build does not fetch whatever is newest.
#
# Nothing but the two programs goes in: they link the C library alone and open
# the display's libraries (EGL, Wayland, X11, xkbcommon) at run time, from the
# system they run on, as every desktop program does.
set -euo pipefail

bin_dir="$1"
version="$2"
out="$3"
: "${APPIMAGETOOL:?the path of appimagetool-x86_64.AppImage}"
: "${APPIMAGE_RUNTIME:?the path of the AppImage runtime-x86_64}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
appdir="$(mktemp -d)/Omnifrog.AppDir"
mkdir -p "$appdir/usr/bin" \
    "$appdir/usr/share/applications" \
    "$appdir/usr/share/icons/hicolor/scalable/apps" \
    "$appdir/usr/share/licenses/omnifrog"

install -m755 "$bin_dir/omnifrog" "$bin_dir/omnifrog-gui" "$appdir/usr/bin/"
install -m755 "$here/AppRun" "$appdir/AppRun"
install -m644 "$here/omnifrog.desktop" "$appdir/usr/share/applications/"
install -m644 "$here/omnifrog.svg" "$appdir/usr/share/icons/hicolor/scalable/apps/"
install -m644 "$root/LICENSE" "$appdir/usr/share/licenses/omnifrog/"
# appimagetool takes the entry and the icon from the AppDir's root.
ln -s usr/share/applications/omnifrog.desktop "$appdir/omnifrog.desktop"
ln -s usr/share/icons/hicolor/scalable/apps/omnifrog.svg "$appdir/omnifrog.svg"
ln -s omnifrog.svg "$appdir/.DirIcon"

# The tool is an AppImage too; extracting it each run spares FUSE, which a
# build machine may not have.
VERSION="$version" APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 \
    "$APPIMAGETOOL" --no-appstream --runtime-file "$APPIMAGE_RUNTIME" "$appdir" "$out"
