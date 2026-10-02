import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _attribute_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_forbidden_framework_apis_are_not_used():
    violations = []
    for path in ROOT.rglob("*.py"):
        if any(part in {".venv", "venv", "__pycache__"} for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "torch.optim":
                allowed = (
                    relative == "src/part2/optimizers.py"
                    and [alias.name for alias in node.names] == ["Optimizer"]
                )
                if not allowed:
                    violations.append(f"{relative}:{node.lineno}: forbidden torch.optim import")
            if isinstance(node, ast.ImportFrom) and node.module in {"torch.nn", "torch"}:
                forbidden_names = {
                    "Transformer", "TransformerEncoder", "TransformerEncoderLayer",
                    "TransformerDecoder", "TransformerDecoderLayer",
                }
                if any(alias.name in forbidden_names for alias in node.names):
                    violations.append(f"{relative}:{node.lineno}: forbidden composite transformer import")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "generate":
                    violations.append(f"{relative}:{node.lineno}: generation API call")
            if isinstance(node, ast.Attribute):
                chain = _attribute_chain(node)
                if chain.startswith("torch.optim."):
                    violations.append(f"{relative}:{node.lineno}: forbidden torch.optim API {chain}")
                if chain.startswith("nn.Transformer") or chain.startswith("torch.nn.Transformer"):
                    violations.append(f"{relative}:{node.lineno}: forbidden composite transformer {chain}")
    assert not violations, "\n".join(violations)