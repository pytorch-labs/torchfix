import libcst as cst
from ...common import TorchVisitor, LintViolation


class TorchUnsafeLoadVisitor(TorchVisitor):
    """
    Warn on `torch.load` not having explicit `weights_only`.
    See https://github.com/pytorch/pytorch/issues/31875.
    """

    ERROR_CODE = "TOR102"
    MESSAGE = (
        "`torch.load` without `weights_only` parameter is unsafe. "
        "Explicitly set `weights_only` to False only if you trust the data you load "
        "and full pickle functionality is needed, otherwise set "
        "`weights_only=True`."
    )

    def visit_Call(self, node):
        qualified_name = self.get_qualified_name_for_call(node)
        if qualified_name == "torch.load":
            weights_only_arg = self.get_specific_arg(node, "weights_only", -1)
            if weights_only_arg is None:
                position_metadata = self.get_metadata(
                    cst.metadata.WhitespaceInclusivePositionProvider, node
                )

                self.violations.append(
                    LintViolation(
                        error_code=self.ERROR_CODE,
                        message=self.MESSAGE,
                        line=position_metadata.start.line,
                        column=position_metadata.start.column,
                        node=node,
                        replacement=None,
                    )
                )
