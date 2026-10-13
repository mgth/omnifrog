; Omnifrog for Windows: the command line and the window, for one user.
;
;   makensis -DVERSION=0.1.0 -DSTAGE=<dir> -DOUTFILE=<setup.exe> omnifrog.nsi
;
; STAGE holds omnifrog.exe, omnifrog-gui.exe, README.md and LICENSE.
;
; Per user, no elevation: the files go under %LOCALAPPDATA%\Programs\Omnifrog,
; with an empty tools\ folder beside them, where the programs look for ffmpeg,
; mkvmerge and the others before PATH. Outside that folder the installer
; writes a Start-menu shortcut and its uninstall entry, nothing else.
;
; The configuration (%APPDATA%\omnifrog) and what `setup` installs
; (%LOCALAPPDATA%\omnifrog: the venv, MSST, the models, the jobs) are the
; program's own: neither the installer nor the uninstaller touches them, and
; the uninstaller leaves whatever was put in tools\.
;
; Switches:
;   /S          install or uninstall without a wizard
;   /D=<dir>    the install directory (last on the line, no quotes)

Unicode true
!include MUI2.nsh
!include FileFunc.nsh

!ifndef VERSION
  !error "pass -DVERSION=<version>"
!endif
!ifndef STAGE
  !error "pass -DSTAGE=<staged files>"
!endif
!ifndef OUTFILE
  !error "pass -DOUTFILE=<installer path>"
!endif

!define PRODUCT "Omnifrog"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\Omnifrog"
!define HOMEPAGE "https://github.com/mgth/omnifrog"

Name "${PRODUCT} ${VERSION}"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Programs\Omnifrog"
; An upgrade goes where the last install went.
InstallDirRegKey HKCU "${UNINST_KEY}" "InstallLocation"
RequestExecutionLevel user
SetCompressor /SOLID lzma
ManifestDPIAware true
ShowInstDetails show
ShowUninstDetails show

; ── Pages ────────────────────────────────────────────────────────────────────
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${STAGE}/LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_TITLE "Omnifrog ${VERSION} is installed"
!define MUI_FINISHPAGE_TEXT "Omnifrog drives tools it does not ship: Python 3.12, ffmpeg, mkvmerge, a decoder and an encoder. The window's Set up page says which are missing and where to put them."
!define MUI_FINISHPAGE_RUN "$INSTDIR\omnifrog-gui.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Open Omnifrog"
!define MUI_FINISHPAGE_LINK "What Omnifrog needs on Windows"
!define MUI_FINISHPAGE_LINK_LOCATION "${HOMEPAGE}#windows"
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

; ── Install ──────────────────────────────────────────────────────────────────
Section "Omnifrog"
  SectionIn RO
  SetShellVarContext current
  SetOutPath "$INSTDIR"
  File "${STAGE}/omnifrog.exe"
  File "${STAGE}/omnifrog-gui.exe"
  File "${STAGE}/README.md"
  File "/oname=LICENSE.txt" "${STAGE}/LICENSE"
  CreateDirectory "$INSTDIR\tools"
  WriteUninstaller "$INSTDIR\uninstall.exe"

  CreateShortcut "$SMPROGRAMS\Omnifrog.lnk" "$INSTDIR\omnifrog-gui.exe"

  WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "${PRODUCT}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\omnifrog-gui.exe"
  WriteRegStr HKCU "${UNINST_KEY}" "URLInfoAbout" "${HOMEPAGE}"
  WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKCU "${UNINST_KEY}" "QuietUninstallString" '"$INSTDIR\uninstall.exe" /S'
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKCU "${UNINST_KEY}" "EstimatedSize" "$0"
SectionEnd

; ── Uninstall ────────────────────────────────────────────────────────────────
; File by file, never the folder as a whole: the tools someone put in tools\
; are theirs, and a folder that still holds something is left where it is.
Section "Uninstall"
  SetShellVarContext current
  Delete "$INSTDIR\omnifrog.exe"
  Delete "$INSTDIR\omnifrog-gui.exe"
  Delete "$INSTDIR\README.md"
  Delete "$INSTDIR\LICENSE.txt"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR\tools"
  RMDir "$INSTDIR"
  Delete "$SMPROGRAMS\Omnifrog.lnk"
  DeleteRegKey HKCU "${UNINST_KEY}"
SectionEnd
