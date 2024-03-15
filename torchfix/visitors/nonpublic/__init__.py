from typing import Sequence

import libcst as cst
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
            self.add_violation(node, error_code=error_code, message=message)

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
                self.add_violation(node, error_code=error_code, message=message)
