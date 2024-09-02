import libcst as cst
import pkgutil
import yaml
from typing import Optional, List

from ...common import (
    TorchVisitor,
    TorchError,
    call_with_name_changes,
    check_old_names_in_import_from,
)

from .range import call_replacement_range
from .cholesky import call_replacement_cholesky
from .chain_matmul import call_replacement_chain_matmul
from .qr import call_replacement_qr


class TorchDeprecatedSymbolsVisitor(TorchVisitor):
    ERRORS: List[TorchError] = [
        TorchError("TOR001", "Use of removed function {old_name}"),
        TorchError("TOR101", "Use of deprecated function {old_name}"),
        TorchError("TOR004", "Import of removed function {old_name}"),
        TorchError("TOR103", "Import of deprecated function {old_name}"),
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
        self.old_new_name_map = {
            name: self.deprecated_config[name].get("replacement")
            for name in self.deprecated_config
        }

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
            return replacements_map[qualified_name](node)

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

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module is None:
            return

        old_names, replacement = check_old_names_in_import_from(
            node, self.old_new_name_map
        )
        for qualified_name in old_names:
            if self.deprecated_config[qualified_name]["remove_pr"] is None:
                error_code = self.ERRORS[3].error_code
                message = self.ERRORS[3].message(old_name=qualified_name)
            else:
                error_code = self.ERRORS[2].error_code
                message = self.ERRORS[2].message(old_name=qualified_name)

            reference = self.deprecated_config[qualified_name].get("reference")
            if reference is not None:
                message = f"{message}: {reference}"

            self.add_violation(
                node,
                error_code=error_code,
                message=message,
                replacement=replacement,
            )

    def visit_Call(self, node) -> None:
        qualified_name = self.get_qualified_name_for_call(node)
        if qualified_name is None:
            return

        if qualified_name in self.deprecated_config:
            if self.deprecated_config[qualified_name]["remove_pr"] is None:
                error_code = self.ERRORS[1].error_code
                message = self.ERRORS[1].message(old_name=qualified_name)
            else:
                error_code = self.ERRORS[0].error_code
                message = self.ERRORS[0].message(old_name=qualified_name)
            replacement = self._call_replacement(node, qualified_name)

            reference = self.deprecated_config[qualified_name].get("reference")
            if reference is not None:
                message = f"{message}: {reference}"

            self.add_violation(
                node, error_code=error_code, message=message, replacement=replacement
            )
