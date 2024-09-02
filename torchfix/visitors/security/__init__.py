import libcst as cst

from ...common import TorchError, TorchVisitor


class TorchUnsafeLoadVisitor(TorchVisitor):
    """
    Warn on `torch.load` not having explicit `weights_only`.
    See https://github.com/pytorch/pytorch/issues/31875.
    """

    ERRORS = [
        TorchError(
            "TOR102",
            (
                "`torch.load` without `weights_only` parameter is unsafe. "
                "Explicitly set `weights_only` to False only if you trust "
                "the data you load "
                "and full pickle functionality is needed,"
                " otherwise set `weights_only=True`."
            ),
        )
    ]

    def visit_Call(self, node):
        if self.get_qualified_name_for_call(
            node
        ) == "torch.load" and not self.has_specific_arg(node, "weights_only"):
            # Add `weights_only=True` if there is no `pickle_module`.
            # (do not add `weights_only=False` with `pickle_module`, as it
            # needs to be an explicit choice).
            #
            # This codemod is somewhat unsafe correctness-wise
            # because full pickling functionality may still be needed
            # even without `pickle_module`,
            # so the changes need to be verified/tested.
            replacement = None
            if not self.has_specific_arg(node, "pickle_module", 2):
                weights_only_arg = cst.ensure_type(
                    cst.parse_expression("f(weights_only=True)"), cst.Call
                ).args[0]
                replacement = node.with_changes(args=(*node.args, weights_only_arg))
            self.add_violation(
                node,
                error_code=self.ERRORS[0].error_code,
                message=self.ERRORS[0].message(),
                replacement=replacement,
            )
