import libcst as cst
from ...common import get_module_name


def call_replacement_chain_matmul(node: cst.Call) -> cst.CSTNode:
    """
    Replace `torch.chain_matmul` with `torch.linalg.multi_dot`, changing
    multiple parameters to a list.
    """
    matrices = [
        cst.Element(value=arg.value) for arg in node.args if arg.keyword is None
    ]
    matrices_arg = cst.Arg(value=cst.List(elements=matrices))

    out_arg = None
    for arg in node.args:
        if arg.keyword is not None and arg.keyword.value == "out":
            out_arg = arg

    replacement_args = [matrices_arg] if out_arg is None else [matrices_arg, out_arg]
    module_name = get_module_name(node, "torch")
    replacement = cst.parse_expression(f"{module_name}.linalg.multi_dot(args)")
    return replacement.with_changes(args=replacement_args)
