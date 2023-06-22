from dataclasses import dataclass
import sys
import libcst as cst
from typing import Optional

IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
CYAN = "\033[96m" if IS_TTY else ""
RED = "\033[31m" if IS_TTY else ""
BOLD = "\033[1m" if IS_TTY else ""
ENDC = "\033[0m" if IS_TTY else ""


@dataclass
class LintViolation:
    error_code: str
    message: str
    line: int
    column: int
    node: cst.CSTNode
    replacement: Optional[cst.CSTNode]

    def flake8_result(self):
        full_message = f"{self.error_code} {self.message}"
        return (self.line, 1 + self.column, full_message, "TorchFix")

    def codemod_result(self) -> str:
        fixable = f" [{CYAN}*{ENDC}]" if self.replacement is not None else ""
        colon = f"{CYAN}:{ENDC}"
        position = f"{colon}{self.line}{colon}{1 + self.column}{colon}"
        error_code = f"{RED}{BOLD}{self.error_code}{ENDC}"
        return f"{position} {error_code}{fixable} {self.message}"


def deep_multi_replace(tree, replacement_map):
    class MultiChildReplacementTransformer(cst.CSTTransformer):
        def __init__(self, replacement_map) -> None:
            self.replacement_map = replacement_map

        def on_leave(self, original_node, updated_node):
            if id(original_node) in self.replacement_map:
                return self.replacement_map[id(original_node)]
            return updated_node

    return tree.visit(MultiChildReplacementTransformer(replacement_map))
