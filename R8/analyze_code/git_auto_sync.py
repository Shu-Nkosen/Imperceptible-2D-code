"""解析終了後に結果を git commit & push する共通ヘルパー。

.gitignore を尊重する（例: out/**/*.png は載らない。CSV は載る）。
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence


def find_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    cur = (start or Path(__file__).resolve()).resolve()
    if cur.is_file():
        cur = cur.parent
    for path in [cur, *cur.parents]:
        if (path / ".git").exists():
            return path
    return None


def _run_git(repo: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def auto_commit_and_push(*, source: str, detail: str = "") -> bool:
    """変更があれば add → commit → push。成功で True。

    失敗しても例外は投げず、ログだけ出して False を返す（解析本体は止めない）。
    """
    repo = find_repo_root()
    if repo is None:
        print("[WARN] auto-git: .git が見つからないため commit/push をスキップ")
        return False

    add = _run_git(repo, ["add", "-A"])
    if add.returncode != 0:
        print(f"[WARN] auto-git: git add 失敗: {(add.stderr or add.stdout).strip()}")
        return False

    status = _run_git(repo, ["status", "--porcelain"])
    if status.returncode != 0:
        print(f"[WARN] auto-git: git status 失敗: {(status.stderr or status.stdout).strip()}")
        return False
    if not (status.stdout or "").strip():
        print("[INFO] auto-git: コミットする変更なし")
        return True

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detail_txt = f" ({detail})" if detail.strip() else ""
    message = f"chore: sync {source} results{detail_txt} @ {stamp}"
    commit = _run_git(repo, ["commit", "-m", message])
    if commit.returncode != 0:
        print(f"[WARN] auto-git: git commit 失敗: {(commit.stderr or commit.stdout).strip()}")
        return False
    print(f"[INFO] auto-git: committed: {message}")

    push = _run_git(repo, ["push"])
    if push.returncode != 0:
        print(f"[WARN] auto-git: git push 失敗: {(push.stderr or push.stdout).strip()}")
        return False
    print("[INFO] auto-git: pushed to remote")
    return True
