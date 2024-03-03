import libcst as cst

from ...common import LintViolation, TorchVisitor


class TorchVisionModelsImportVisitor(TorchVisitor):
    ERROR_CODE = "TOR203"

    def visit_Import(self, node: cst.Import) -> None:
        for imported_item in node.names:
            if isinstance(imported_item.name, cst.Attribute):
                if (
                    isinstance(imported_item.name.value, cst.Name)
                    and imported_item.name.value.value == "torchvision"
                    and imported_item.name.attr.value == "models"
                    and imported_item.asname is not None
                    and imported_item.asname.name.value == "models"
                ):
                    print(imported_item.asname.name.value)
                    position = self.get_metadata(
                        cst.metadata.WhitespaceInclusivePositionProvider, node
                    )
                    # print(position)
                    replacement = cst.ImportFrom(
                            module=cst.Name("torchvision"),
                            names=[cst.ImportAlias(name=cst.Name("models"))],
                            )
                    self.violations.append(
                        LintViolation(
                            error_code=self.ERROR_CODE,
                            message=(
                                "Consider replacing 'import torchvision.models as"
                                " models' with 'from torchvision import models'. "
                            ),
                            line=position.start.line,
                            column=position.start.column,
                            node=node,
                            replacement=replacement
                        )
                    )
