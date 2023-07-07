import libcst as cst
import libcst.matchers as m
from typing import Optional, Tuple


def call_replacement_range(node: cst.Call) -> Optional[cst.Call]:
    """Replace `range` with `arange`.
    Add `step` to the `end` argument as `arange` has the interval `[start, end)`.
    """
    # `torch.range` documented signature is not a valid Python signature,
    # so it's hard to generalize this.
    def _get_range_args(node: cst.Call) -> Tuple[cst.Arg, Optional[cst.Arg]]:
        "Return (`end`, `step`) from a `range` call"
        end_arg = None
        step_arg = None
        non_kw_args = []
        for arg in node.args:
            if arg.keyword is None:
                non_kw_args.append(arg)
            else:
                if arg.keyword.value == "end":
                    end_arg = arg
                elif arg.keyword.value == "step":
                    step_arg = arg
        if end_arg is None:
            if len(non_kw_args) == 1:
                end_arg = non_kw_args[0]
            elif len(non_kw_args) == 2:
                end_arg = non_kw_args[1]
            elif len(non_kw_args) == 3:
                end_arg = non_kw_args[1]
                step_arg = non_kw_args[2]
        assert isinstance(end_arg, cst.Arg)
        return end_arg, step_arg

    end_arg, step_arg = _get_range_args(node)
    step = 1
    if step_arg is not None:
        # `step` is a literal integer
        if isinstance(step_arg.value, cst.Integer):
            step = int(step_arg.value.value)

        # `step` is unary minus and an integer (i.e. negative integer)
        elif m.matches(
            step_arg,
            m.Arg(value=m.UnaryOperation(operator=m.Minus(), expression=m.Integer())),
        ):
            # Ignore type error here and further in this file.
            # See https://github.com/Instagram/LibCST/issues/964
            step = -int(step_arg.value.expression.value)  # type: ignore

        # Bail out, don't know how to update with non-integer `step`.
        else:
            return None

    updated_end_arg = None

    # `end` is a literal (positive) integer
    if isinstance(end_arg.value, cst.Integer):
        end = int(end_arg.value.value) + step
        if end >= 0:
            updated_end_arg = end_arg.with_deep_changes(
                old_node=end_arg.value, value=str(end)
            )
        else:
            # `end` became negative
            updated_end_arg = end_arg.with_changes(
                value=cst.UnaryOperation(
                    operator=cst.Minus(),
                    expression=cst.Integer(value=str(-end)),
                )
            )

    # `end` is a unary minus and an integer (i.e. negative integer)
    elif m.matches(
        end_arg,
        m.Arg(value=m.UnaryOperation(operator=m.Minus(), expression=m.Integer())),
    ):
        end = -int(end_arg.value.expression.value) + step  # type: ignore
        if end < 0:
            updated_end_arg = end_arg.with_deep_changes(
                old_node=end_arg.value.expression, value=str(-end)  # type: ignore
            )
        else:
            # `end` became non-negative
            updated_end_arg = end_arg.with_changes(value=cst.Integer(value=str(end)))

    # `end` is an expression with `- 1` at the end: remove the `- 1`.
    # This is a common occurrence, thus special handling.
    elif m.matches(
        end_arg,
        m.Arg(
            value=m.BinaryOperation(operator=m.Subtract(), right=m.Integer(value="1"))
        ),
    ):
        updated_end_arg = end_arg.with_changes(value=end_arg.value.left)  # type: ignore

    # `end` something else: add `+ 1` at the end
    else:
        updated_end_arg = end_arg.with_changes(
            value=cst.BinaryOperation(
                left=end_arg.value,
                operator=cst.Add(),
                right=cst.Integer(value="1"),
            )
        )

    replacement = node
    if updated_end_arg is not None:
        # Ignore type error, see https://github.com/Instagram/LibCST/issues/965
        replacement = replacement.deep_replace(end_arg, updated_end_arg)  # type: ignore
    replacement = replacement.with_deep_changes(
        old_node=cst.ensure_type(replacement.func, cst.Attribute).attr, value="arange"
    )

    return replacement
