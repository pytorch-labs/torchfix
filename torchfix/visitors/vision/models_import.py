import libcst as cst
import libcst.matchers as m

from ...common import TorchVisitor


class TorchVisionModelsImportVisitor(TorchVisitor):
    ERROR_CODE = "TOR203"
    MESSAGE = (
        "Consider replacing 'import torchvision.models as models' "
        "with 'from torchvision import models'."
    )

    def visit_Import(self, node: cst.Import) -> None:
        replacement = None
        for imported_item in node.names:
            if m.matches(imported_item, m.ImportAlias(
                name=m.Attribute(value=m.Name("torchvision"),
                                attr=m.Name("models")),
                asname=m.AsName(name=m.Name("models"))
            )):
                # Replace only if the import statement has no other names
                if len(node.names) == 1:
                    replacement = cst.ImportFrom(
                        module=cst.Name("torchvision"),
                        names=[cst.ImportAlias(name=cst.Name("models"))],
                    )
                self.add_violation(
                    node,
                    error_code=self.ERROR_CODE,
                    message=self.MESSAGE,
                    replacement=replacement,
                )
            break
