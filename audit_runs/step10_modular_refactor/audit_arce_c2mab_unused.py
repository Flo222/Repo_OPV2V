import ast
from pathlib import Path
from collections import Counter

path = Path("opencood/comm/arce/arce_c2mab_comm.py")
tree = ast.parse(path.read_text(encoding="utf-8"))

imports = []
defined_funcs = []
defined_classes = []
assigned_names = []
used_names = []

class Visitor(ast.NodeVisitor):
    def visit_Import(self, node):
        for a in node.names:
            imports.append((a.asname or a.name.split(".")[0], a.name, node.lineno))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        mod = node.module or ""
        for a in node.names:
            imports.append((a.asname or a.name, f"{mod}.{a.name}", node.lineno))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        defined_funcs.append((node.name, node.lineno))
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        defined_classes.append((node.name, node.lineno))
        self.generic_visit(node)

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Name):
                assigned_names.append((t.id, node.lineno))
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            used_names.append(node.id)
        self.generic_visit(node)

Visitor().visit(tree)

used = Counter(used_names)

print("===== IMPORTS possibly unused =====")
for local, src, lineno in imports:
    if used[local] == 0:
        print(f"L{lineno}: {local} from {src}")

print("\n===== TOP-LEVEL FUNCTIONS possibly unused =====")
for name, lineno in defined_funcs:
    if used[name] <= 1:
        print(f"L{lineno}: {name} used_count={used[name]}")

print("\n===== TOP-LEVEL CONSTANTS possibly unused =====")
for name, lineno in assigned_names:
    if name.isupper() and used[name] <= 1:
        print(f"L{lineno}: {name} used_count={used[name]}")

print("\n===== CLASSES =====")
for name, lineno in defined_classes:
    print(f"L{lineno}: {name}")
