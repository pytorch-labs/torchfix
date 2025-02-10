"""size_average and reduce are deprecated, please use reduction='mean' instead."""

import libcst as cst
from ...common import TorchVisitor, get_module_name
from torch.nn._reduction import legacy_get_string

def call_replacement_loss(node: cst.Call) -> cst.CSTNode:
    """
    Replace loss function that contains size_average / reduce with a new loss function 
    that uses reduction='mean' instead. Uses the logic from torch.nn._reduction to 
    determine the correct reduction value.
    
    Args:
        node: The CST Call node representing the loss function call
        
    Returns:
        A new CST node with updated reduction parameter
    """
    # Extract existing arguments
    input_arg = TorchVisitor.get_specific_arg(node, "input", 0)
    target_arg = TorchVisitor.get_specific_arg(node, "target", 1)

    size_average_arg = TorchVisitor.get_specific_arg(node, "size_average", 2)
    reduce_arg = TorchVisitor.get_specific_arg(node, "reduce", 3)
    
    # Ensure input and target args maintain their commas
    input_arg = cst.ensure_type(input_arg, cst.Arg).with_changes(
        comma=cst.MaybeSentinel.DEFAULT
    )

    target_arg = cst.ensure_type(target_arg, cst.Arg).with_changes(
        comma=cst.MaybeSentinel.DEFAULT
    )

    # Extract size_average and reduce values
    size_average_value = None
    reduce_value = None
    
    if size_average_arg:
        size_average_value = getattr(size_average_arg.value, "value", True)
    if reduce_arg:
        reduce_value = getattr(reduce_arg.value, "value", True)
    
    if size_average_value is None and reduce_value is None:
        # We want to return the original call as is
        return node
    # Use legacy_get_string to determine the correct reduction value
    reduction = legacy_get_string(size_average_value, reduce_value, emit_warning=False)
    
    # Create new reduction argument
    reduction_arg = cst.Arg(
        value=cst.SimpleString(f"'{reduction}'"),
        keyword=cst.Name("reduction"),
        comma=cst.MaybeSentinel.DEFAULT,
    )
    
    # Build new arguments list
    new_args = [input_arg, target_arg, reduction_arg]
    replacement = node.with_changes(args=new_args)
    return replacement
