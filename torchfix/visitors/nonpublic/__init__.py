from typing import List

import libcst as cst

from ...common import (
    TorchError,
    TorchVisitor,
    call_with_name_changes,
    check_old_names_in_import_from,
)


class TorchNonPublicAliasVisitor(TorchVisitor):
    """
    Suggest to use public APIs instead of non-public aliases.

    Currently implemented for
    torch.utils.data._utils.collate.default_collate and
    torch.utils.data._utils.collate.default_convert,
    see https://github.com/pytorch/pytorch/pull/69862/files
    """

    ERRORS: List[TorchError] = [
        TorchError(
            "TOR104",
            (
                "Use of non-public function `{private_name}`, "
                "please use `{public_name}` instead"
            ),
        ),
        TorchError(
            "TOR105",
            (
                "Import of non-public function `{private_name}`, "
                "please use `{public_name}` instead"
            ),
        ),
    ]

    # fmt: off
    ALIASES = {
        "torch.utils.data._utils.collate.default_collate": "torch.utils.data.dataloader.default_collate",  # noqa: E501
        "torch.utils.data._utils.collate.default_convert": "torch.utils.data.dataloader.default_convert",  # noqa: E501
    }
    # fmt: on

    def visit_Call(self, node):
        qualified_name = self.get_qualified_name_for_call(node)
        if qualified_name is None:
            return

        if qualified_name in self.ALIASES:
            public_name = self.ALIASES[qualified_name]
            error_code = self.ERRORS[0].error_code
            message = self.ERRORS[0].message(
                private_name=qualified_name, public_name=public_name
            )

            replacement_and_imports = call_with_name_changes(
                node, qualified_name, public_name
            )
            if replacement_and_imports is not None:
                replacement, imports = replacement_and_imports
                self.needed_imports.update(imports)
            else:
                replacement = None

            self.add_violation(
                node, error_code=error_code, message=message, replacement=replacement
            )

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module is None:
            return

        private_names, replacement = check_old_names_in_import_from(node, self.ALIASES)
        for qualified_name in private_names:
            public_name = self.ALIASES[qualified_name]
            error_code = self.ERRORS[1].error_code
            message = self.ERRORS[1].message(
                private_name=qualified_name, public_name=public_name
            )
            self.add_violation(
                node,
                error_code=error_code,
                message=message,
                replacement=replacement,
            )
