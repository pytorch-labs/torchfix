import libcst as cst
import libcst.matchers as m

from ...common import TorchError, TorchVisitor


class TorchRequireGradVisitor(TorchVisitor):
    """
    Find and fix common misspelling `require_grad` (instead of `requires_grad`).
    """

    ERRORS = [
        TorchError(
            "TOR002",
            "Likely typo `require_grad` in assignment. Did you mean `requires_grad`?",
        )
    ]

    def visit_Assign(self, node):
        # Look for any assignment with `require_grad` attribute on the left.
        #
        # This is unlikely to cause false-positives on real code, especially
        # because TorchFix only looks at files that have a `torch` string.
        if m.matches(
            node,
            m.Assign(
                targets=[
                    m.AssignTarget(
                        target=m.Attribute(attr=m.Name(value="require_grad"))
                    )
                ],
            ),
        ):
            replacement = node.with_deep_changes(
                old_node=node.targets[0].target.attr, value="requires_grad"
            )
            self.add_violation(
                node,
                error_code=self.ERRORS[0].error_code,
                message=self.ERRORS[0].message(),
                replacement=replacement,
            )


class TorchReentrantCheckpointVisitor(TorchVisitor):
    """
    Find and fix common misuse of reentrant checkpoints.
    """

    ERRORS = [
        TorchError(
            "TOR003",
            (
                "Please pass `use_reentrant` explicitly to `checkpoint`. "
                "To maintain old behavior, pass `use_reentrant=True`. "
                "It is recommended to use `use_reentrant=False`."
            ),
        )
    ]

    def visit_Call(self, node):
        if self.get_qualified_name_for_call(
            node
        ) == "torch.utils.checkpoint.checkpoint" and not self.has_specific_arg(
            node, "use_reentrant"
        ):
            # This codemod maybe  unsafe correctness-wise
            # if reentrant behavior is actually needed,
            # so the changes need to be verified/tested.
            use_reentrant_arg = cst.ensure_type(
                cst.parse_expression("f(use_reentrant=False)"), cst.Call
            ).args[0]
            replacement = node.with_changes(args=(*node.args, use_reentrant_arg))
            self.add_violation(
                node,
                error_code=self.ERRORS[0].error_code,
                message=self.ERRORS[0].message(),
                replacement=replacement,
            )


class TorchLog1pVisitor(TorchVisitor):
    """
    Suggest using `torch.log1p(x)` instead of `torch.log(1 + x)`.
    """

    ERRORS = [
        TorchError(
            "TOR106",
            (
                "Use `torch.log1p(x)` instead of `torch.log(1 + x)`. "
                "It is more accurate for small values of `x`."
            ),
        )
    ]

    def visit_Call(self, node):
        if self.get_qualified_name_for_call(node) == "torch.log":

            if m.matches(
                node,
                m.Call(
                    args=[
                        m.Arg(
                            value=m.BinaryOperation(
                                left=m.Integer(value="1") | m.Float(value="1.0"),
                                operator=m.Add(),
                            )
                            | m.BinaryOperation(
                                operator=m.Add(),
                                right=m.Integer(value="1") | m.Float(value="1.0"),
                            ),
                        ),
                    ],
                ),
            ):

                self.add_violation(
                    node,
                    error_code=self.ERRORS[0].error_code,
                    message=self.ERRORS[0].message(),
                    replacement=None,
                )


class TorchExpm1Visitor(TorchVisitor):
    """
    Suggest using `torch.special.expm1(x)` instead of `torch.exp(x) - 1`.
    """

    ERRORS = [
        TorchError(
            "TOR107",
            (
                "Use `torch.special.expm1(x)` instead of `torch.exp(x) - 1`. "
                "It is more accurate for small values of `x`."
            ),
        )
    ]

    def visit_BinaryOperation(self, node):
        if m.matches(
            node,
            m.BinaryOperation(
                left=m.Call(),
                operator=m.Subtract(),
                right=m.Integer(value="1") | m.Float(value="1.0"),
            ),
        ):
            if self.get_qualified_name_for_call(node.left) == "torch.exp":
                self.add_violation(
                    node,
                    error_code=self.ERRORS[0].error_code,
                    message=self.ERRORS[0].message(),
                    replacement=None,
                )
