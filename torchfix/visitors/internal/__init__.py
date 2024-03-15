from ...common import TorchVisitor


class TorchScopedLibraryVisitor(TorchVisitor):
    """
    Suggest `torch.library._scoped_library` for PyTorch tests.
    """

    ERROR_CODE = "TOR901"
    MESSAGE = (
        "Use `torch.library._scoped_library` instead of `torch.library.Library` "
        "in PyTorch tests files. See https://github.com/pytorch/pytorch/pull/118318 "
        "for details."
    )

    def visit_Call(self, node):
        qualified_name = self.get_qualified_name_for_call(node)
        if qualified_name == "torch.library.Library":
            self.add_violation(node, error_code=self.ERROR_CODE, message=self.MESSAGE)
