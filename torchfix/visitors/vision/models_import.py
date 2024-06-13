import libcst as cst
import libcst.matchers as m

from ...common import TorchError, TorchVisitor


# TODO (jovaun) change file name and corresponding fixtures
class TorchVisionModelsImportVisitor(TorchVisitor):
    ERRORS = [
        TorchError(
            "TOR203",
            (
                "Consider replacing 'import torchvision.datasets as datasets' "
                "with 'from torchvision import datasets'."
            ),
        ),
        TorchError(
            "TOR203",
            (
                "Consider replacing 'import torchvision.models as models' "
                "with 'from torchvision import models'."
            ),
        ),
        TorchError(
            "TOR203",
            (
                "Consider replacing 'import torchvision.transforms as transforms' "
                "with 'from torchvision import transforms'."
            ),
        ),
    ]

    # Keep attr order in sync with ERRORS.
    REPLACEABLE_ATTRS = ["datasets", "models", "transforms"]

    def visit_Import(self, node: cst.Import) -> None:
        replacement = None
        for i, import_attr in enumerate(self.REPLACEABLE_ATTRS):
            for imported_item in node.names:
                if m.matches(
                    imported_item,
                    m.ImportAlias(
                        name=m.Attribute(
                            value=m.Name("torchvision"), attr=m.Name(import_attr)
                        ),
                        asname=m.AsName(name=m.Name(import_attr)),
                    ),
                ):
                    # Replace only if the import statement has no other names
                    if len(node.names) == 1:
                        replacement = cst.ImportFrom(
                            module=cst.Name("torchvision"),
                            names=[cst.ImportAlias(name=cst.Name(import_attr))],
                        )
                    self.add_violation(
                        node,
                        error_code=self.ERRORS[i].error_code,
                        message=self.ERRORS[i].message(),
                        replacement=replacement,
                    )
                break
