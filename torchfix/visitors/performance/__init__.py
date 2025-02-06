import libcst.matchers as m

from ...common import TorchError, TorchVisitor


class TorchSynchronizedDataLoaderVisitor(TorchVisitor):
    """
    Reimplementation of SynchronizedDataLoaderPattern from
    https://github.com/pytorch/pytorch/blob/main/torch/profiler/_pattern_matcher.py
    """

    ERRORS = [
        TorchError(
            "TOR401",
            (
                "Detected DataLoader running with synchronized implementation."
                " Please enable asynchronous dataloading by setting "
                "num_workers > 0 when initializing DataLoader."
            ),
        )
    ]

    def visit_Call(self, node):
        qualified_name = self.get_qualified_name_for_call(node)
        if qualified_name == "torch.utils.data.DataLoader":
            num_workers_arg = self.get_specific_arg(node, "num_workers", 5)
            if num_workers_arg is None or m.matches(
                num_workers_arg.value, m.Integer(value="0")
            ):
                self.add_violation(
                    node,
                    error_code=self.ERRORS[0].error_code,
                    message=self.ERRORS[0].message(),
                )


class TorchGradNotSetToNonePatternVisitor(TorchVisitor):
    """
    Reimplementation of GradNotSetToNonePattern from
    https://github.com/pytorch/pytorch/blob/main/torch/profiler/_pattern_matcher.py
    """

    ERRORS = [
        TorchError(
            "TOR402",
            (
                "Detected gradient set to zero instead of None. "
                "Please add 'set_to_none=True' when calling zero_grad()."
            ),
        )
    ]

    def visit_Call(self, node):
        qualified_name = self.get_qualified_name_for_call(node)

        if qualified_name and qualified_name.endswith("zero_grad"):

            set_to_none_arg = self.get_specific_arg(node, "set_to_none", 0)

            # hasattr check to handle mypy error
            if set_to_none_arg and hasattr(set_to_none_arg.value, "value"):
                if set_to_none_arg.value.value == "False":
                    self.add_violation(
                        node,
                        error_code=self.ERRORS[0].error_code,
                        message=self.ERRORS[0].message(),
                    )
