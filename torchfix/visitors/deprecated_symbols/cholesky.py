import libcst as cst
from ...common import (TorchVisitor, get_module_name)


def call_replacement_cholesky(node: cst.Call) -> cst.CSTNode:
    """
    Replace `torch.cholesky(A)` with `torch.linalg.cholesky(A)` and
    `torch.cholesky(A, upper=True)` with `torch.linalg.cholesky(A).mH`.
    """
    input_arg = TorchVisitor.get_specific_arg(node, "input", 0)
    input_arg = cst.ensure_type(input_arg, cst.Arg).with_changes(
        comma=cst.MaybeSentinel.DEFAULT
    )
    upper_arg = TorchVisitor.get_specific_arg(node, "upper", 1)
    module_name = get_module_name(node, "torch")

    if (
        upper_arg is not None
        and cst.ensure_type(upper_arg.value, cst.Name).value == "True"
    ):
        replacement = cst.parse_expression(f"{module_name}.linalg.cholesky(A).mH")
        replacement = replacement.with_deep_changes(
            # Ignore type error, see https://github.com/Instagram/LibCST/issues/963
            old_node=cst.ensure_type(replacement.value, cst.Call).args,  # type: ignore
            value=[input_arg],
        )
    else:
        replacement = cst.parse_expression(f"{module_name}.linalg.cholesky(A)")
        replacement = replacement.with_changes(args=[input_arg])

    return replacement
