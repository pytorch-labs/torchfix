import libcst as cst
from typing import Optional
from ...common import (TorchVisitor, get_module_name)


def call_replacement_qr(node: cst.Call) -> Optional[cst.CSTNode]:
    """
    Replace `torch.qr(A)` with `torch.linalg.qr(A)` and
    `torch.qr(A, some=False)` with `torch.linalg.qr(A, mode="complete")`.
    """
    input_arg = TorchVisitor.get_specific_arg(node, "input", 0)
    if input_arg is None:
        return None

    some_arg = TorchVisitor.get_specific_arg(node, "some", 1)

    if (
        some_arg is not None
        and cst.ensure_type(some_arg.value, cst.Name).value == "False"
    ):
        mode_arg = cst.ensure_type(
            cst.parse_expression('f(mode="complete")'), cst.Call
        ).args[0]
        replacement_args = [input_arg, mode_arg]
    else:
        input_arg = cst.ensure_type(input_arg, cst.Arg).with_changes(
            comma=cst.MaybeSentinel.DEFAULT
        )
        replacement_args = [input_arg]
    module_name = get_module_name(node, "torch")
    replacement = cst.parse_expression(f"{module_name}.linalg.qr(args)")
    replacement = replacement.with_changes(args=replacement_args)

    return replacement
