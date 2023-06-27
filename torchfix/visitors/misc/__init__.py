import libcst as cst
import libcst.matchers as m


from ...common import TorchVisitor, LintViolation


class TorchRequireGradVisitor(TorchVisitor):
    """
    Find and fix common misspelling `require_grad` (instead of `requires_grad`).
    """

    ERROR_CODE = "TOR002"
    MESSAGE = "Likely typo `require_grad` in assignment. Did you mean `requires_grad`?"

    def visit_Assign(self, node):
        # Look for any assignment with `require_grad` attribute on the left
        # and `False` or `True` on the right.
        #
        # If this causes false-positives on real code (unlikely),
        # we can do type inference (not sure if feasible here) or
        # at least check that `torch` is imported in the file.
        if m.matches(
            node,
            m.Assign(
                targets=[
                    m.AssignTarget(
                        target=m.Attribute(attr=m.Name(value="require_grad"))
                    )
                ],
                value=(m.Name("True") | m.Name("False")),
            ),
        ):
            replacement = node.with_deep_changes(
                old_node=node.targets[0].target.attr, value="requires_grad"
            )

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
                    replacement=replacement,
                )
            )
