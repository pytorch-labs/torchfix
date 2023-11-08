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
