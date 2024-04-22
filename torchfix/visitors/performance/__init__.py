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
