import libcst as cst
import pkgutil
import yaml
from typing import Optional, List

import libcst as cst
from libcst.metadata import QualifiedNameProvider

from ...common import (
    TorchError,
    TorchVisitor,
    call_with_name_changes,
    check_old_names_in_import_from,
)
from .chain_matmul import call_replacement_chain_matmul
from .cholesky import call_replacement_cholesky
from .qr import call_replacement_qr
from .range import call_replacement_range


class TorchDeprecatedSymbolsVisitor(TorchVisitor):
    ERRORS: List[TorchError] = [
        TorchError("TOR001", "Use of removed function {old_name}"),
        TorchError("TOR101", "Use of deprecated function {old_name}"),
        TorchError("TOR004", "Import of removed function {old_name}"),
        TorchError("TOR103", "Import of deprecated function {old_name}"),
        TorchError("TOR005", "Reference to removed function {old_name}"),
        TorchError("TOR106", "Reference to deprecated function {old_name}"),
    ]

    def __init__(self, deprecated_config_path=None):
        def read_deprecated_config(path=None):
            deprecated_config = {}
            if path is not None:
                data = pkgutil.get_data("torchfix", path)
                assert data is not None
                for item in yaml.load(data, yaml.SafeLoader):
                    deprecated_config[item["name"]] = item
            return deprecated_config

        super().__init__()
        self.deprecated_config = read_deprecated_config(deprecated_config_path)
        self.replacements = {
            name: self.deprecated_config[name].get("replacement")
            for name in self.deprecated_config
        }
        self.in_call = False

    def _call_replacement(
        self, node: cst.Call, qualified_name: str
    ) -> Optional[cst.CSTNode]:
        replacements_map = {
            "torch.cholesky": call_replacement_cholesky,
            "torch.range": call_replacement_range,
            "torch.chain_matmul": call_replacement_chain_matmul,
            "torch.qr": call_replacement_qr,
        }

        if qualified_name in replacements_map:
            return replacements_map[qualified_name](node)

        replacement = None

        # Replace names for functions that have drop-in replacement.
        function_name_replacement = self.deprecated_config.get(qualified_name, {}).get(
            "replacement", ""
        )
        if function_name_replacement:
            replacement_and_imports = call_with_name_changes(
                node, qualified_name, function_name_replacement
            )
            if replacement_and_imports is not None:
                replacement, imports = replacement_and_imports
                self.needed_imports.update(imports)
        return replacement

    def _construct_error(self, qualified_name, deprecated_key, removed_key):
        if "remove_pr" not in self.deprecated_config[qualified_name]:
            error_code = self.ERRORS[deprecated_key].error_code
            message = self.ERRORS[deprecated_key].message(old_name=qualified_name)
        else:
            error_code = self.ERRORS[removed_key].error_code
            message = self.ERRORS[removed_key].message(old_name=qualified_name)

        reference = self.deprecated_config[qualified_name].get("reference")
        if reference is not None:
            message = f"{message}: {reference}"

        return error_code, message

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module is None:
            return

        old_names, replacement = check_old_names_in_import_from(node, self.replacements)
        for qualified_name in old_names:
            error_code, message = self._construct_error(qualified_name, 3, 2)

            self.add_violation(
                node,
                error_code=error_code,
                message=message,
                replacement=replacement,
            )

    def visit_Call(self, node: cst.Call) -> None:
        self.in_call = True
        qualified_name = self.get_qualified_name_for_call(node)
        if qualified_name is None:
            return

        if qualified_name not in self.deprecated_config:
            return

        error_code, message = self._construct_error(qualified_name, 1, 0)
        replacement = self._call_replacement(node, qualified_name)
        self.add_violation(
            node, error_code=error_code, message=message, replacement=replacement
        )

    def leave_Call(self, original_node: cst.Call) -> None:
        self.in_call = False

    def visit_Attribute(self, node: cst.Attribute):
        # avoid duplicates
        if self.in_call:
            return False

        name_metadata = list(self.get_metadata(QualifiedNameProvider, node))
        if not name_metadata:
            return False

        qualified_name = name_metadata[0].name
        if qualified_name not in self.deprecated_config:
            return False

        error_code, message = self._construct_error(qualified_name, 4, 5)
        self.add_violation(node, error_code=error_code, message=message)
        return None
