import sys
from abc import ABC
from dataclasses import dataclass
from os.path import commonprefix
from typing import List, Optional, Sequence, Set, Tuple, Mapping

import libcst as cst
from libcst.codemod.visitors import ImportItem
from libcst.metadata import QualifiedNameProvider, WhitespaceInclusivePositionProvider

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
        return self.line, 1 + self.column, full_message, "TorchFix"

    def codemod_result(self) -> str:
        fixable = f" [{CYAN}*{ENDC}]" if self.replacement is not None else ""
        colon = f"{CYAN}:{ENDC}"
        position = f"{colon}{self.line}{colon}{1 + self.column}{colon}"
        error_code = f"{RED}{BOLD}{self.error_code}{ENDC}"
        return f"{position} {error_code}{fixable} {self.message}"


@dataclass(frozen=True)
class TorchError:
    """Defines an error along with an explanation"""

    error_code: str
    message_template: str

    def message(self, **kwargs):
        return self.message_template.format(**kwargs)


class TorchVisitor(cst.BatchableCSTVisitor, ABC):
    METADATA_DEPENDENCIES = (
        QualifiedNameProvider,
        WhitespaceInclusivePositionProvider,
    )

    ERRORS: List[TorchError]

    def __init__(self) -> None:
        super().__init__()
        self.violations: List[LintViolation] = []
        self.needed_imports: Set[ImportItem] = set()

    @staticmethod
    def get_specific_arg(
        node: cst.Call, arg_name: str, arg_pos: int
    ) -> Optional[cst.Arg]:
        """
        :param arg_pos: `arg_pos` is zero-based. -1 means it's a keyword argument.
        :note: consider using `has_specific_arg` if you only need to check for presence.
        """
        curr_pos = 0
        for arg in node.args:
            if arg.keyword is None:
                if curr_pos == arg_pos:
                    return arg
                curr_pos += 1
            elif arg.keyword.value == arg_name:
                return arg
        return None

    @staticmethod
    def has_specific_arg(
        node: cst.Call, arg_name: str, position: Optional[int] = None
    ) -> bool:
        """
        Check if the specific argument is present in a call.
        """
        return (
            TorchVisitor.get_specific_arg(
                node, arg_name, position if position is not None else -1
            )
            is not None
        )

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
        return name_metadata[0].name


def call_with_name_changes(
    node: cst.Call, qualified_name: str, new_qualified_name: str
) -> Optional[Tuple[cst.Call, Set[ImportItem]]]:
    """
    Return an optional tuple:
    new `Call` node with name changes
    and a set of newly needed imports.
    """
    needed_imports: Set[ImportItem] = set()
    call_name = cst.helpers.get_full_name_for_node(node)
    assert call_name is not None
    replacement = None

    alias_prefix = ""
    if not qualified_name.endswith(call_name):
        # This means we have an alias (`import from as`).
        common_suffix = commonprefix([qualified_name[::-1], call_name[::-1]])[::-1]
        alias_prefix = call_name.removesuffix(common_suffix) + "."

    if not new_qualified_name.endswith(call_name):
        # We need to change the call name as it's not a part of the new qualified name.
        # Get the new call name on the same hierarchical level.
        new_call_name = new_qualified_name.removeprefix(
            commonprefix([qualified_name.removesuffix(call_name), new_qualified_name])
        )
        new_module_name = new_qualified_name.removesuffix(new_call_name).removesuffix(
            "."
        )
        if new_module_name:
            needed_imports.add(
                ImportItem(
                    module_name=new_module_name,
                    obj_name=new_call_name.split(".")[0],
                )
            )
        replacement = node.with_changes(
            func=cst.parse_expression(alias_prefix + new_call_name)
        )

    # Replace with new_qualified_name.
    if replacement is None:
        return None

    return replacement, needed_imports


def check_old_names_in_import_from(
    node: cst.ImportFrom, old_new_name_map: Mapping[str, Optional[str]]
) -> Tuple[List[str], Optional[cst.ImportFrom]]:
    """
    Using `old_new_name_map`, check if there are any old names in the import from.
    Return a tuple of two elements:
    1. List of all founds old names.
    2. Optional replacement for the ImportFrom node.
    """
    if node.module is None or not isinstance(node.names, Sequence):
        return [], None

    old_names: List[str] = []
    replacement = None
    new_names: List[str] = []
    module = cst.helpers.get_full_name_for_node(node.module)

    # `possible_new_modules` and `has_non_updated_names` are used
    # to decide if we can replace the ImportFrom node.
    new_modules: Set[str] = set()
    has_non_updated_names = False

    for name in node.names:
        qualified_name = f"{module}.{name.name.value}"
        if qualified_name in old_new_name_map:
            old_names.append(qualified_name)
            new_qualified_name = old_new_name_map[qualified_name]
            if new_qualified_name is not None:
                new_module = ".".join(new_qualified_name.split(".")[:-1])
                new_name = new_qualified_name.split(".")[-1]
                new_names.append(new_name)
                new_modules.add(new_module)
            else:
                has_non_updated_names = True
        else:
            has_non_updated_names = True

    # Replace only if the new module is the same for all names in the import.
    if not has_non_updated_names and len(new_modules) == 1:
        new_module = new_modules.pop()
        import_aliases = [
            import_alias.with_changes(name=cst.Name(new_name))
            for import_alias, new_name in zip(list(node.names), new_names)
        ]
        replacement = node.with_changes(
            module=cst.parse_expression(new_module),
            names=import_aliases,
        )

    return old_names, replacement


def deep_multi_replace(tree, replacement_map):
    class MultiChildReplacementTransformer(cst.CSTTransformer):
        def __init__(self, replacement_map) -> None:
            super().__init__()
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
