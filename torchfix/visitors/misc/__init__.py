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
