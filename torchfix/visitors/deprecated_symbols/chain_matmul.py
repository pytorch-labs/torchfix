import libcst as cst
from ...common import get_module_name


def call_replacement_chain_matmul(node: cst.Call) -> cst.CSTNode:
    """
    Replace `torch.chain_matmul` with `torch.linalg.multi_dot`, changing
    multiple parameters to a list.
    """
    matrices = []
    out_arg = None
    for arg in node.args:
        if arg.keyword is None:
            matrices.append(cst.Element(value=arg.value))
        elif arg.keyword.value == "out":
            out_arg = arg
    matrices_arg = cst.Arg(value=cst.List(elements=matrices))

    if out_arg is None:
        replacement_args = [matrices_arg]
    else:
        replacement_args = [matrices_arg, out_arg]
    module_name = get_module_name(node, 'torch')
    replacement = cst.parse_expression(f"{module_name}.linalg.multi_dot(args)")
    replacement = replacement.with_changes(args=replacement_args)

    return replacement
