from dataclasses import dataclass
import sys
import libcst as cst
from libcst.metadata import QualifiedNameProvider, WhitespaceInclusivePositionProvider
from libcst.codemod.visitors import ImportItem
from typing import Optional, List, Set, Tuple, Union
from abc import ABC

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


class TorchVisitor(cst.BatchableCSTVisitor, ABC):
    METADATA_DEPENDENCIES = (
        QualifiedNameProvider,
        WhitespaceInclusivePositionProvider,
    )

    ERROR_CODE: Union[str, List[str]]

    def __init__(self) -> None:
        self.violations: List[LintViolation] = []
        self.needed_imports: Set[ImportItem] = set()

    @staticmethod
    def get_specific_arg(
        node: cst.Call, arg_name: str, arg_pos: int
    ) -> Optional[cst.Arg]:
        # `arg_pos` is zero-based.
        curr_pos = 0
        for arg in node.args:
            if arg.keyword is None:
                if curr_pos == arg_pos:
                    return arg
                curr_pos += 1
            elif arg.keyword.value == arg_name:
                return arg
        return None

    def add_violation(
        self,
        node: cst.CSTNode,
        error_code: str,
        message: str,
        replacement: Optional[cst.CSTNode] = None,
    ) -> None:
        position_metadata = self.get_metadata(
            cst.metadata.WhitespaceInclusivePositionProvider, node
        )
        self.violations.append(
            LintViolation(
                error_code=error_code,
                message=message,
                line=position_metadata.start.line,
                column=position_metadata.start.column,
                node=node,
                replacement=replacement,
            )
        )

    def get_qualified_name_for_call(self, node: cst.Call) -> Optional[str]:
        # Guard against situations like `vmap(a)(b)`:
        #
        # Call(
        #   func=Call(
        #       func=Name(
        #         value='vmap',
        #
        # The QualifiedName metadata for the outer call will be the same
        # as for the inner call.
        if isinstance(node.func, cst.Call):
            return None

        name_metadata = list(self.get_metadata(QualifiedNameProvider, node))
        if not name_metadata:
            return None
        qualified_name = name_metadata[0].name
        return qualified_name


def call_with_name_changes(
    node: cst.Call, old_qualified_name: str, new_qualified_name: str
) -> Optional[Tuple[cst.Call, Set[ImportItem]]]:
    """
    Return an optional tuple:
    new `Call` node with name changes
    and a set of newly needed imports.
    """
    old_begin, _, old_last = old_qualified_name.rpartition(".")
    new_begin, _, new_last = new_qualified_name.rpartition(".")
    needed_imports: Set[ImportItem] = set()

    # If the only difference is the last name part.
    if old_begin == new_begin:
        if isinstance(node.func, cst.Attribute):
            replacement = node.with_deep_changes(
                old_node=node.func.attr,
                value=new_last,
            )
        elif isinstance(node.func, cst.Name):
            replacement = node.with_deep_changes(
                old_node=node.func,
                value=new_last,
            )
            needed_imports.add(
                ImportItem(
                    module_name=new_begin,
                    obj_name=new_last,
                )
            )

    # If the last name part is the same and
    # originally called without a dot: don't change the call site,
    # just change the imports elsewhere.
    elif old_last == new_last and isinstance(node.func, cst.Name):
        replacement = None

    # Replace with new_qualified_name.
    else:
        replacement = node.with_changes(func=cst.parse_expression(new_qualified_name))
    if replacement is None:
        return None
    else:
        return replacement, needed_imports


def deep_multi_replace(tree, replacement_map):
    class MultiChildReplacementTransformer(cst.CSTTransformer):
        def __init__(self, replacement_map) -> None:
            self.replacement_map = replacement_map

        def on_leave(self, original_node, updated_node):
            if id(original_node) in self.replacement_map:
                return self.replacement_map[id(original_node)]
            return updated_node

    return tree.visit(MultiChildReplacementTransformer(replacement_map))


def get_module_name(node: cst.Call, default: Optional[str] = None) -> Optional[str]:
    if not isinstance(node.func, cst.Attribute):
        return default
    if not isinstance(node.func.value, cst.Name):
        return default
    return node.func.value.value
