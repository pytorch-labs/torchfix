from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import libcst as cst
import libcst.codemod as codemod

from .common import deep_multi_replace
from .visitors.deprecated_symbols import (
    TorchDeprecatedSymbolsVisitor,
    _UpdateFunctorchImports,
)

from .visitors.performance import TorchSynchronizedDataLoaderVisitor
from .visitors.misc import (TorchRequireGradVisitor, TorchReentrantCheckpointVisitor)

from .visitors.vision import (
    TorchVisionDeprecatedPretrainedVisitor,
    TorchVisionDeprecatedToTensorVisitor,
)
from .visitors.security import TorchUnsafeLoadVisitor

__version__ = "0.3.0"

DEPRECATED_CONFIG_PATH = Path(__file__).absolute().parent / "deprecated_symbols.yaml"

DISABLED_BY_DEFAULT = ["TOR3", "TOR4"]

ALL_VISITOR_CLS = [
    TorchDeprecatedSymbolsVisitor,
    TorchRequireGradVisitor,
    TorchSynchronizedDataLoaderVisitor,
    TorchVisionDeprecatedPretrainedVisitor,
    TorchVisionDeprecatedToTensorVisitor,
    TorchUnsafeLoadVisitor,
    TorchReentrantCheckpointVisitor,
]

_ALL_ERROR_CODES = None


def GET_ALL_ERROR_CODES():
    global _ALL_ERROR_CODES
    if _ALL_ERROR_CODES is None:
        codes = []
        for cls in ALL_VISITOR_CLS:
            if isinstance(cls.ERROR_CODE, list):
                codes += cls.ERROR_CODE
            else:
                codes.append(cls.ERROR_CODE)
        _ALL_ERROR_CODES = codes
    return _ALL_ERROR_CODES


def GET_ALL_VISITORS():
    out = []
    for v in ALL_VISITOR_CLS:
        if v is TorchDeprecatedSymbolsVisitor:
            out.append(v(DEPRECATED_CONFIG_PATH))
        else:
            out.append(v())
    return out


def get_visitor_with_error_code(error_code):
    # Each error code can only correspond to one visitor
    for visitor in GET_ALL_VISITORS():
        if isinstance(visitor.ERROR_CODE, list):
            if error_code in visitor.ERROR_CODE:
                return visitor
        else:
            if error_code == visitor.ERROR_CODE:
                return visitor
    raise AssertionError(f"Unknown error code: {error_code}")


def get_visitors_with_error_codes(error_codes):
    visitors = []
    for error_code in error_codes:
        visitors.append(get_visitor_with_error_code(error_code))
    return visitors


# Flake8 plugin
class TorchChecker:
    name = "TorchFix"
    version = __version__

    # The parameters need to have these exact names.
    # See https://flake8.pycqa.org/en/latest/plugin-development/plugin-parameters.html
    # `tree` is unused, but the plugin doesn't work without it.
    def __init__(self, tree, lines):
        # Filter out files that don't have "torch" string in them.
        # This avoids expensive parsing.
        MARKER = "torch"  # this will catch import torch or functorch
        has_marker = False
        self.module = None
        for line in lines:
            if MARKER in line:
                has_marker = True
                break
        if has_marker:
            module = cst.parse_module("".join(lines))
            self.module = cst.MetadataWrapper(module, unsafe_skip_copy=True)
            self.violations = []
            self.visitors = GET_ALL_VISITORS()

    def run(self):
        if self.module:
            self.module.visit_batched(self.visitors)
            for v in self.visitors:
                self.violations += v.violations
            for violation in self.violations:
                yield violation.flake8_result()

    @staticmethod
    def add_options(optmanager):
        optmanager.extend_default_ignore(DISABLED_BY_DEFAULT)


# Standalone torchfix command
@dataclass
class TorchCodemodConfig:
    select: Optional[List[str]] = None


class TorchCodemod(codemod.Codemod):
    def __init__(
        self,
        context: codemod.CodemodContext,
        config: Optional[TorchCodemodConfig] = None,
    ) -> None:
        super().__init__(context)
        self.config = config

    def transform_module_impl(self, module: cst.Module) -> cst.Module:
        # We use `unsafe_skip_copy`` here not only to save some time, but
        # because `deep_replace`` is identity-based and will not work on
        # the original module if the wrapper does a deep copy:
        # in that case we would need to use `wrapped_module.module`
        # instead of `module`.
        wrapped_module = cst.MetadataWrapper(module, unsafe_skip_copy=True)
        if self.config is None or self.config.select is None:
            raise AssertionError("Expected self.config.select to be set")
        visitors = get_visitors_with_error_codes(self.config.select)

        violations = []
        needed_imports = []
        wrapped_module.visit_batched(visitors)
        for v in visitors:
            violations += v.violations
            needed_imports += v.needed_imports

        fixes_count = 0
        replacement_map = {}
        assert self.context.filename is not None
        for violation in violations:
            # Still need to skip violations here, since a single visitor can
            # correspond to multiple different types of violations.
            skip_violation = True
            for code in self.config.select:
                if violation.error_code.startswith(code):
                    skip_violation = False
                    break
            if skip_violation:
                continue

            if violation.replacement is not None:
                replacement_map[id(violation.node)] = violation.replacement
                fixes_count += 1
            try:
                path = Path(self.context.filename).relative_to(Path.cwd())
            except ValueError:
                # Not a subpath of a current dir, use absolute path
                path = Path(self.context.filename)
            print(f"{path}{violation.codemod_result()}")

        new_module = deep_multi_replace(module, replacement_map)

        add_imports_visitor = codemod.visitors.AddImportsVisitor(
            self.context, needed_imports
        )
        new_module = new_module.visit(add_imports_visitor)

        update_functorch_imports_visitor = _UpdateFunctorchImports()
        new_module = new_module.visit(update_functorch_imports_visitor)

        if fixes_count == 0 and not update_functorch_imports_visitor.changed:
            raise codemod.SkipFile("No changes")

        return new_module
