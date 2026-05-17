@echo off
setlocal EnableDelayedExpansion

echo ============================================
echo   Git LFS Push Helper for .pth model files
echo ============================================
echo.

:: ── Check git is available ──────────────────────────────────────────────────
where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] git is not installed or not in PATH.
    exit /b 1
)

:: ── Check git lfs is available ──────────────────────────────────────────────
where git-lfs >nul 2>&1
if errorlevel 1 (
    echo [ERROR] git-lfs is not installed or not in PATH.
    echo         Download it from https://git-lfs.com
    exit /b 1
)

:: ── Check we are inside a git repo ──────────────────────────────────────────
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Not inside a Git repository.
    echo         Run this script from anywhere inside your repo.
    exit /b 1
)

:: ── cd to repo root so recursive search always covers everything ─────────────
for /f "delims=" %%R in ('git rev-parse --show-toplevel') do set REPO_ROOT=%%R
echo       Repo root: %REPO_ROOT%
cd /d "%REPO_ROOT%"

:: ── Initialise LFS (safe to run multiple times) ─────────────────────────────
echo [1/5] Initialising Git LFS...
git lfs install >nul 2>&1
echo       Done.
echo.

:: ── Track *.pth if not already tracked ──────────────────────────────────────
echo [2/5] Tracking *.pth files with Git LFS...
git lfs track "*.pth" >nul 2>&1
echo       Done.
echo.

:: ── Stage .gitattributes ─────────────────────────────────────────────────────
echo [3/5] Staging .gitattributes...
git add .gitattributes
echo       Done.
echo.

:: ── Find and stage every .pth file in the repo ──────────────────────────────
echo [4/5] Staging all .pth files...
set found=0
for /r "%REPO_ROOT%" %%F in (*.pth) do (
    set found=1
    echo       + %%F
    git add "%%F"
)
if !found!==0 (
    echo       No .pth files found in this repo.
)
echo.

:: ── Commit (skip if nothing new to commit) ──────────────────────────────────
git diff --cached --quiet
if errorlevel 1 (
    echo [4/5] Committing staged files...
    git commit -m "chore: track .pth model files via Git LFS"
    echo.
) else (
    echo [4/5] Nothing new to commit - working tree is clean.
    echo.
)

:: ── Push ────────────────────────────────────────────────────────────────────
echo [5/5] Pushing to remote (including LFS objects)...
echo.
git push
if errorlevel 1 (
    echo.
    echo [ERROR] Push failed. Common causes:
    echo         - Not authenticated with GitHub
    echo         - Remote is ahead of local  ^(run git pull first^)
    echo         - GitHub LFS storage quota exceeded
    exit /b 1
)

echo.
echo ============================================
echo   All done! .pth files pushed via LFS.
echo ============================================