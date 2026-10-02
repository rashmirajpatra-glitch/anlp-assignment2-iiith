from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "data",
    "outputs",
    "runs",
    "wandb",
}
EXCLUDED_FILES = {"uv.lock"}


def _candidate_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(ROOT).parts
        if any(part in EXCLUDED_DIRECTORIES for part in relative_parts):
            continue
        if path.name in EXCLUDED_FILES:
            continue
        yield path


def test_source_and_assets_stay_under_repository_size_limits():
    files = list(_candidate_files())
    oversized = [
        f"{path.relative_to(ROOT)} ({path.stat().st_size} bytes)"
        for path in files
        if path.stat().st_size > 300 * 1024
    ]
    assert not oversized, "Files exceed 300 KB:\n" + "\n".join(oversized)
    total_bytes = sum(path.stat().st_size for path in files)
    assert total_bytes < 5 * 1024 * 1024, f"Source/assets total {total_bytes} bytes exceeds 5 MB"