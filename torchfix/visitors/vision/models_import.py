import libcst as cst

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
            if isinstance(imported_item.name, cst.Attribute):
                # TODO refactor using libcst.matchers.matches
                if (
                    isinstance(imported_item.name.value, cst.Name)
                    and imported_item.name.value.value == "torchvision"
                    and isinstance(imported_item.name.attr, cst.Name)
                    and imported_item.name.attr.value == "models"
                    and imported_item.asname is not None
                    and isinstance(imported_item.asname.name, cst.Name)
                    and imported_item.asname.name.value == "models"
                ):
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
