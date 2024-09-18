from dataclasses import dataclass
import functools
from pathlib import Path
from typing import Optional, List
import libcst as cst
import libcst.codemod as codemod

from .common import deep_multi_replace, TorchVisitor

from .visitors import (
    TorchDeprecatedSymbolsVisitor,
    TorchExpm1Visitor,
    TorchLog1pVisitor,
    TorchNonPublicAliasVisitor,
    TorchReentrantCheckpointVisitor,
    TorchRequireGradVisitor,
    TorchScopedLibraryVisitor,
    TorchSynchronizedDataLoaderVisitor,
    TorchUnsafeLoadVisitor,
    TorchVisionDeprecatedPretrainedVisitor,
    TorchVisionDeprecatedToTensorVisitor,
    TorchVisionSingletonImportVisitor,
)

__version__ = "0.6.0"

DEPRECATED_CONFIG_PATH = "deprecated_symbols.yaml"

DISABLED_BY_DEFAULT = ["TOR3", "TOR4", "TOR9"]

ALL_VISITOR_CLS = [
    TorchDeprecatedSymbolsVisitor,
    TorchExpm1Visitor,
    TorchLog1pVisitor,
    TorchNonPublicAliasVisitor,
    TorchRequireGradVisitor,
    TorchReentrantCheckpointVisitor,
    TorchScopedLibraryVisitor,
    TorchSynchronizedDataLoaderVisitor,
    TorchUnsafeLoadVisitor,
    TorchVisionDeprecatedPretrainedVisitor,
    TorchVisionDeprecatedToTensorVisitor,
    TorchVisionSingletonImportVisitor,
]


@functools.cache
def GET_ALL_ERROR_CODES():
    codes = set()
    for cls in ALL_VISITOR_CLS:
        assert issubclass(cls, TorchVisitor)
        codes |= {error.error_code for error in cls.ERRORS}
    return sorted(codes)


@functools.cache
def expand_error_codes(codes):
    out_codes = set()
    for c_a in codes:
        for c_b in GET_ALL_ERROR_CODES():
            if c_b.startswith(c_a):
                out_codes.add(c_b)
    return out_codes


def construct_visitor(cls):
    if cls is TorchDeprecatedSymbolsVisitor:
        return cls(DEPRECATED_CONFIG_PATH)

    return cls()


def GET_ALL_VISITORS():
    return [construct_visitor(v) for v in ALL_VISITOR_CLS]


def get_visitors_with_error_codes(error_codes):
    visitor_classes = set()
    for error_code in error_codes:
        # Assume the error codes have been expanded so each error code can
        # only correspond to one visitor.
        found = False
        for visitor_cls in ALL_VISITOR_CLS:
            assert issubclass(visitor_cls, TorchVisitor)
            if any(error_code == err.error_code for err in visitor_cls.ERRORS):
                visitor_classes.add(visitor_cls)
                found = True
                break
        if not found:
            raise AssertionError(f"Unknown error code: {error_code}")
    return [construct_visitor(cls) for cls in visitor_classes]


def process_error_code_str(code_str):
    # Allow duplicates in the input string, e.g. --select ALL,TOR0,TOR001.
    # We deduplicate them here.

    # Default when --select is not provided.
    if code_str is None:
        exclude_set = expand_error_codes(tuple(DISABLED_BY_DEFAULT))
        return set(GET_ALL_ERROR_CODES()) - exclude_set

    raw_codes = [s.strip() for s in code_str.split(",")]

    # Validate error codes
    for c in raw_codes:
        if c == "ALL":
            continue
        if len(expand_error_codes((c,))) == 0:
            raise ValueError(
                f"Invalid error code: {c}, available error "
                f"codes: {list(GET_ALL_ERROR_CODES())}"
            )

    if "ALL" in raw_codes:
        return GET_ALL_ERROR_CODES()

    return expand_error_codes(tuple(raw_codes))


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

        if fixes_count == 0:
            raise codemod.SkipFile("No changes")

        return new_module
