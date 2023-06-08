from pathlib import Path
import yaml
import libcst as cst
import libcst.codemod as codemod
import libcst.matchers as m
from dataclasses import dataclass
from typing import Optional, List, Tuple
import argparse
import contextlib
import io
import sys

__version__ = "0.0.1"

IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
CYAN = "\033[96m" if IS_TTY else ""
RED = "\033[31m" if IS_TTY else ""
BOLD = "\033[1m" if IS_TTY else ""
ENDC = "\033[0m" if IS_TTY else ""


@dataclass
class LintViolation:
    error_code: str
    message: str
    line: int
    column: int
    node: cst.CSTNode
    replacement: Optional[cst.CSTNode]

    def flake8_result(self):
        full_message = f"{self.error_code} {self.message}"
        return (self.line, 1 + self.column, full_message, "TorchFix")

    def codemod_result(self) -> str:
        fixable = f" [{CYAN}*{ENDC}]" if self.replacement is not None else ""
        colon = f"{CYAN}:{ENDC}"
        position = f"{colon}{self.line}{colon}{1 + self.column}{colon}"
        error_code = f"{RED}{BOLD}{self.error_code}{ENDC}"
        return f"{position} {error_code}{fixable} {self.message}"


class _MultiChildReplacementTransformer(cst.CSTTransformer):
    def __init__(self, replacement_map) -> None:
        self.replacement_map = replacement_map

    def on_leave(self, original_node, updated_node):
        if id(original_node) in self.replacement_map:
            return self.replacement_map[id(original_node)]
        return updated_node


def deep_multi_replace(tree, replacement_map):
    return tree.visit(_MultiChildReplacementTransformer(replacement_map))


class TorchVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        cst.metadata.QualifiedNameProvider,
        cst.metadata.WhitespaceInclusivePositionProvider,
    )

    def _call_replacement(self, node: cst.Call, qualified_name) -> cst.Call:

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

            return end_arg, step_arg

        replacement = None
        if qualified_name == "torch.ger":
            replacement = node.with_deep_changes(old_node=node.func.attr, value="outer")

        # Replace `range` with `arange`.
        # Add `step` to the `end` argument as `arange` has the interval `[start, end)`.
        if qualified_name == "torch.range":
            end_arg, step_arg = _get_range_args(node)
            step = 1
            if step_arg is not None:
                # `step` is a literal integer
                if isinstance(step_arg.value, cst.Integer):
                    step = int(step_arg.value.value)

                # `step` is unary minus and an integer (i.e. negative integer)
                elif m.matches(
                    step_arg,
                    m.Arg(
                        value=m.UnaryOperation(
                            operator=m.Minus(), expression=m.Integer()
                        )
                    ),
                ):
                    step = -int(step_arg.value.expression.value)

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
                m.Arg(
                    value=m.UnaryOperation(operator=m.Minus(), expression=m.Integer())
                ),
            ):
                end = -int(end_arg.value.expression.value) + step
                if end < 0:
                    updated_end_arg = end_arg.with_deep_changes(
                        old_node=end_arg.value.expression, value=str(-end)
                    )
                else:
                    # `end` became non-negative
                    updated_end_arg = end_arg.with_changes(
                        value=cst.Integer(value=str(end))
                    )

            # `end` is an expression with `- 1` at the end: remove the `- 1`.
            # This is a common occurrence, thus special handling.
            elif m.matches(
                end_arg,
                m.Arg(
                    value=m.BinaryOperation(
                        operator=m.Subtract(), right=m.Integer(value="1")
                    )
                ),
            ):
                updated_end_arg = end_arg.with_changes(value=end_arg.value.left)

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
                replacement = replacement.deep_replace(end_arg, updated_end_arg)
            replacement = replacement.with_deep_changes(
                old_node=replacement.func.attr, value="arange"
            )

        return replacement

    def __init__(self, deprecated_config=None):
        self.deprecated_config = {} if deprecated_config is None else deprecated_config
        self.violations: List[LintViolation] = []

    def visit_Call(self, node):
        name_metadata = list(
            self.get_metadata(cst.metadata.QualifiedNameProvider, node)
        )
        if not name_metadata:
            return
        qualified_name = name_metadata[0].name

        if qualified_name in self.deprecated_config:
            position_metadata = self.get_metadata(
                cst.metadata.WhitespaceInclusivePositionProvider, node
            )
            if self.deprecated_config[qualified_name]["remove_pr"] is None:
                error_code = "TOR101"
                message = f"Use of deprecated function {qualified_name}"
            else:
                error_code = "TOR001"
                message = f"Use of removed function {qualified_name}"

            replacement = self._call_replacement(node, qualified_name)

            self.violations.append(
                LintViolation(
                    error_code=error_code,
                    message=message,
                    line=position_metadata.start.line,
                    column=position_metadata.start.column,
                    node=node,
                    replacement=replacement,
                )
            )


def _read_deprecated_config(path=None):
    if path is None:
        path = Path(__file__).absolute().parent / "deprecated_symbols.yaml"

    deprecated_config = {}
    with open(path) as f:
        for item in yaml.load(f, yaml.SafeLoader):
            deprecated_config[item["name"]] = item
    return deprecated_config


class TorchChecker:
    name = "TorchFix"
    version = __version__

    # The parameters need to have these exact names.
    # See https://flake8.pycqa.org/en/latest/plugin-development/plugin-parameters.html
    # `tree` is unused, but the plugin doesn't work without it.
    def __init__(self, tree, lines):
        module = cst.parse_module("".join(lines))
        visitor = TorchVisitor(_read_deprecated_config())
        module = cst.MetadataWrapper(module, unsafe_skip_copy=True)
        module.visit(visitor)
        self.violations = visitor.violations

    def run(self):
        for violation in self.violations:
            yield violation.flake8_result()


class TorchCodemod(codemod.Codemod):
    def transform_module_impl(self, module: cst.Module) -> cst.Module:
        visitor = TorchVisitor(_read_deprecated_config())
        fixes_count = 0

        # We use `unsafe_skip_copy`` here not only to save some time, but
        # because `deep_replace`` is identity-based and will not work on
        # the original module if the wrapper does a deep copy:
        # in that case we would need to use `wrapped_module.module`
        # instead of `module`.
        wrapped_module = cst.MetadataWrapper(module, unsafe_skip_copy=True)
        wrapped_module.visit(visitor)

        violations = visitor.violations
        replacement_map = {}
        for violation in violations:
            if violation.replacement is not None:
                replacement_map[id(violation.node)] = violation.replacement
                fixes_count += 1
            try:
                path = Path(self.context.filename).relative_to(Path.cwd())
            except ValueError:
                # Not a subpath of a current dir, use absolute path
                path = self.context.filename
            print(f"{path}{violation.codemod_result()}")
        if fixes_count == 0:
            raise codemod.SkipFile("No changes")

        new_module = deep_multi_replace(module, replacement_map)
        return new_module


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "path",
        nargs="+",
        help=("Path to check/fix. Can be a directory, a file, or multiple of either."),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix fixable violations.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        help="Number of jobs to use when processing files. Defaults to number of cores",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    files = codemod.gather_files(args.path)
    command_instance = TorchCodemod(codemod.CodemodContext())
    DIFF_CONTEXT = 5
    try:
        # XXX TODO: Get rid of this!
        # Silence "Failed to determine module name"
        # https://github.com/Instagram/LibCST/issues/944
        with contextlib.redirect_stderr(io.StringIO()):
            result = codemod.parallel_exec_transform_with_prettyprint(
                command_instance,
                files,
                jobs=args.jobs,
                unified_diff=(None if args.fix else DIFF_CONTEXT),
                hide_progress=True,
                format_code=False,
                repo_root=None,
            )
    except KeyboardInterrupt:
        print("Interrupted!", file=sys.stderr)
        sys.exit(2)

    print(
        f"Finished checking {result.successes + result.skips + result.failures} files.",
        file=sys.stderr,
    )

    if result.successes > 0:
        if args.fix:
            print(
                f"Transformed {result.successes} files successfully.", file=sys.stderr
            )
        else:
            print(
                f"[{CYAN}*{ENDC}] {result.successes} "
                "potentially fixable with the --fix option",
                file=sys.stderr,
            )

    if result.failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
