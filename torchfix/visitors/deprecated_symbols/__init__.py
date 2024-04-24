import libcst as cst
import pkgutil
import yaml
from typing import Optional, List
from collections.abc import Sequence

from ...common import (
    TorchVisitor,
    TorchError,
    call_with_name_changes,
)

from .range import call_replacement_range
from .cholesky import call_replacement_cholesky
from .chain_matmul import call_replacement_chain_matmul
from .qr import call_replacement_qr


class TorchDeprecatedSymbolsVisitor(TorchVisitor):
    ERRORS: List[TorchError] = [
        TorchError("TOR001", "Use of removed function {qualified_name}"),
        TorchError("TOR101", "Use of deprecated function {qualified_name}"),
        TorchError("TOR004", "Import of removed function {qualified_name}"),
        TorchError("TOR103", "Import of deprecated function {qualified_name}"),
    ]

    def __init__(self, deprecated_config_path=None):
        def read_deprecated_config(path=None):
            deprecated_config = {}
            if path is not None:
                data = pkgutil.get_data("torchfix", path)
                for item in yaml.load(data, yaml.SafeLoader):
                    deprecated_config[item["name"]] = item
            return deprecated_config

        super().__init__()
        self.deprecated_config = read_deprecated_config(deprecated_config_path)

    def _call_replacement(
        self, node: cst.Call, qualified_name: str
    ) -> Optional[cst.CSTNode]:
        replacements_map = {
            "torch.cholesky": call_replacement_cholesky,
            "torch.range": call_replacement_range,
            "torch.chain_matmul": call_replacement_chain_matmul,
            "torch.qr": call_replacement_qr,
        }
        replacement = None

        if qualified_name in replacements_map:
            replacement = replacements_map[qualified_name](node)
        else:
            # Replace names for functions that have drop-in replacement.
            function_name_replacement = self.deprecated_config.get(
                qualified_name, {}
            ).get("replacement", "")
            if function_name_replacement:
                replacement_and_imports = call_with_name_changes(
                    node, qualified_name, function_name_replacement
                )
                if replacement_and_imports is not None:
                    replacement, imports = replacement_and_imports
                    self.needed_imports.update(imports)
        return replacement

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module is None:
            return

        module = cst.helpers.get_full_name_for_node(node.module)
        if isinstance(node.names, Sequence):
            for name in node.names:
                qualified_name = f"{module}.{name.name.value}"
                if qualified_name in self.deprecated_config:
                    if self.deprecated_config[qualified_name]["remove_pr"] is None:
                        error_code = self.ERRORS[3].error_code
                        message = self.ERRORS[3].message(qualified_name=qualified_name)
                    else:
                        error_code = self.ERRORS[2].error_code
                        message = self.ERRORS[2].message(qualified_name=qualified_name)

                    reference = self.deprecated_config[qualified_name].get("reference")
                    if reference is not None:
                        message = f"{message}: {reference}"

                    self.add_violation(node, error_code=error_code, message=message)

    def visit_Call(self, node) -> None:
        qualified_name = self.get_qualified_name_for_call(node)
        if qualified_name is None:
            return

        if qualified_name in self.deprecated_config:
            if self.deprecated_config[qualified_name]["remove_pr"] is None:
                error_code = self.ERRORS[1].error_code
                message = self.ERRORS[1].message(qualified_name=qualified_name)
            else:
                error_code = self.ERRORS[0].error_code
                message = self.ERRORS[0].message(qualified_name=qualified_name)
            replacement = self._call_replacement(node, qualified_name)

            reference = self.deprecated_config[qualified_name].get("reference")
            if reference is not None:
                message = f"{message}: {reference}"

            self.add_violation(
                node, error_code=error_code, message=message, replacement=replacement
            )


# TODO: refactor/generalize this.
class _UpdateFunctorchImports(cst.CSTTransformer):
    REPLACEMENTS = {
        "vmap",
        "grad",
        "vjp",
        "jvp",
        "jacrev",
        "jacfwd",
        "hessian",
        "functionalize",
    }

    def __init__(self):
        self.changed = False

    def leave_ImportFrom(
        self, node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        if (
            getattr(node.module, "value", None) == "functorch"
            and isinstance(node.names, Sequence)
            and all(name.name.value in self.REPLACEMENTS for name in node.names)
        ):
            self.changed = True
            return updated_node.with_changes(module=cst.parse_expression("torch.func"))
        return updated_node
