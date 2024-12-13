import libcst as cst

from ...common import get_module_name


def call_replacement_cpu_amp_autocast(node: cst.Call) -> cst.CSTNode:
    return _call_replacement_amp(node, "cpu")


def call_replacement_cuda_amp_autocast(node: cst.Call) -> cst.CSTNode:
    return _call_replacement_amp(node, "cuda")


def _call_replacement_amp(node: cst.Call, device: str) -> cst.CSTNode:
    """
    Replace `torch.cuda.amp.autocast()` with `torch.amp.autocast("cuda")` and
    Replace `torch.cpu.amp.autocast()` with `torch.amp.autocast("cpu")`.
    """
    device_arg = cst.ensure_type(cst.parse_expression(f'f("{device}")'), cst.Call).args[
        0
    ]

    module_name = get_module_name(node, "torch")
    replacement = cst.parse_expression(f"{module_name}.amp.autocast(args)")
    replacement = replacement.with_changes(args=(device_arg, *node.args))
    return replacement
