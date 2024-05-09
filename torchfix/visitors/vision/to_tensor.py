from collections.abc import Sequence

import libcst as cst

from ...common import TorchError, TorchVisitor

MESSAGE = (
    "The transform `v2.ToTensor()` is deprecated and will be removed "
    "in a future release. Instead, please use "
    "`v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])`."  # noqa: E501
)


class TorchVisionDeprecatedToTensorVisitor(TorchVisitor):
    ERRORS = [TorchError("TOR202", MESSAGE)]

    def _maybe_add_violation(self, qualified_name, node):
        if qualified_name != "torchvision.transforms.v2.ToTensor":
            return
        self.add_violation(
            node, error_code=self.ERRORS[0].error_code, message=self.ERRORS[0].message()
        )

    def visit_ImportFrom(self, node):
        module_path = cst.helpers.get_absolute_module_from_package_for_import(
            None, node
        )
        if module_path is None:
            return

        if isinstance(node.names, Sequence):
            for import_node in node.names:
                self._maybe_add_violation(
                    f"{module_path}.{import_node.evaluated_name}", import_node
                )

    def visit_Attribute(self, node):
        qualified_names = self.get_metadata(cst.metadata.QualifiedNameProvider, node)
        if len(qualified_names) != 1:
            return

        self._maybe_add_violation(list(qualified_names)[0].name, node)
