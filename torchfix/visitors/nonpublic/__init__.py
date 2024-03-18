from os.path import commonprefix
from typing import Sequence

import libcst as cst
from libcst.codemod.visitors import ImportItem

from ...common import TorchVisitor


class TorchNonPublicAliasVisitor(TorchVisitor):
    """
    Suggest to use public APIs instead of non-public aliases.

    Currently implemented for
    torch.utils.data._utils.collate.default_collate and
    torch.utils.data._utils.collate.default_convert,
    see https://github.com/pytorch/pytorch/pull/69862/files
    """

    ERROR_CODE = ["TOR104", "TOR105"]

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
            error_code = self.ERROR_CODE[0]
            message = f"Use of non-public function `{qualified_name}`, please use `{public_name}` instead"  # noqa: E501

            call_name = cst.helpers.get_full_name_for_node(node)
            replacement = None
            if not public_name.endswith(call_name):
                # We need to change the call name as it's not in the public name.
                # Get the new call name on the same hierarchical level.
                new_call_name = public_name.removeprefix(
                    commonprefix([qualified_name.removesuffix(call_name), public_name])
                )
                new_module_name = public_name.removesuffix(new_call_name).removesuffix(
                    "."
                )
                if new_module_name:
                    self.needed_imports.add(
                        ImportItem(
                            module_name=new_module_name,
                            obj_name=new_call_name.split(".")[0],
                        )
                    )
                replacement = node.with_changes(
                    func=cst.parse_expression(new_call_name)
                )

            self.add_violation(
                node, error_code=error_code, message=message, replacement=replacement
            )

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module is None:
            return

        module = cst.helpers.get_full_name_for_node(node.module)
        if not isinstance(node.names, Sequence):
            return

        for name in node.names:
            qualified_name = f"{module}.{name.name.value}"
            if qualified_name in self.ALIASES:
                public_name = self.ALIASES[qualified_name]
                error_code = self.ERROR_CODE[1]
                message = f"Import of non-public function `{qualified_name}`, please use `{public_name}` instead"  # noqa: E501

                new_module = ".".join(public_name.split(".")[:-1])
                new_name = public_name.split(".")[-1]
                # Replace only if the import statement has no other names
                if len(node.names) == 1:
                    replacement = cst.ImportFrom(
                        module=cst.parse_expression(new_module),  # type: ignore[arg-type] # noqa: E501
                        names=[cst.ImportAlias(name=cst.Name(new_name))],
                    )
                else:
                    replacement = None
                self.add_violation(
                    node,
                    error_code=error_code,
                    message=message,
                    replacement=replacement,
                )
