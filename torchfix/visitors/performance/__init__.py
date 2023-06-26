import libcst as cst
import libcst.matchers as m


from ...common import TorchVisitor, LintViolation


class TorchSynchronizedDataLoaderVisitor(TorchVisitor):
    """
    Reimplementation of SynchronizedDataLoaderPattern from
    https://github.com/pytorch/pytorch/blob/main/torch/profiler/_pattern_matcher.py
    """

    ERROR_CODE = "TOR401"
    MESSAGE = (
        "Detected DataLoader running with synchronized implementation. "
        "Please enable asynchronous dataloading by setting num_workers > 0 when "
        "initializing DataLoader."
    )

    def visit_Call(self, node):
        qualified_name = self.get_qualified_name_for_call(node)
        if qualified_name == "torch.utils.data.DataLoader":
            num_workers_arg = self.get_specific_arg(node, "num_workers", 5)
            if num_workers_arg is None or m.matches(
                num_workers_arg.value, m.Integer(value="0")
            ):
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
