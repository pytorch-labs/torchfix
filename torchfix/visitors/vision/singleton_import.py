import libcst as cst
import libcst.matchers as m

from ...common import TorchError, TorchVisitor


class TorchVisionSingletonImportVisitor(TorchVisitor):
    ERRORS = [
        TorchError(
            "TOR203",
            (
                "Consider replacing 'import torchvision.{module} as {module}' "
                "with 'from torchvision import {module}'."
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
                        error_code=self.ERRORS[0].error_code,
                        message=self.ERRORS[0].message(module=import_attr),
                        replacement=replacement,
                    )
                break
