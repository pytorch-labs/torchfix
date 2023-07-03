import libcst as cst
from ...common import TorchVisitor


def call_replacement_cholesky(node: cst.Call) -> cst.CSTNode:
    """
    Replace `torch.cholesky(A)` with `torch.linalg.cholesky(A)` and
    `torch.cholesky(A, upper=True)` with `torch.linalg.cholesky(A).mH`.
    """
    input_arg = TorchVisitor.get_specific_arg(node, "input", 0).with_changes(
        comma=cst.MaybeSentinel.DEFAULT
    )
    upper_arg = TorchVisitor.get_specific_arg(node, "upper", 1)

    if upper_arg is not None and upper_arg.value.value == "True":
        replacement = cst.parse_expression("torch.linalg.cholesky(A).mH")
        replacement = replacement.with_deep_changes(
            old_node=replacement.value.args, value=[input_arg]
        )
    else:
        replacement = cst.parse_expression("torch.linalg.cholesky(A)").with_changes(
            args=[input_arg]
        )

    return replacement
