from pathlib import Path
import libcst as cst
import libcst.codemod as codemod

from .common import deep_multi_replace
from .visitors.deprecated_symbols import (
    TorchDeprecatedSymbolsVisitor,
    _UpdateFunctorchImports,
)

from .visitors.performance import TorchSynchronizedDataLoaderVisitor

__version__ = "0.0.2"

DEPRECATED_CONFIG_PATH = Path(__file__).absolute().parent / "deprecated_symbols.yaml"


class TorchChecker:
    name = "TorchFix"
    version = __version__

    # The parameters need to have these exact names.
    # See https://flake8.pycqa.org/en/latest/plugin-development/plugin-parameters.html
    # `tree` is unused, but the plugin doesn't work without it.
    def __init__(self, tree, lines):
        module = cst.parse_module("".join(lines))
        self.module = cst.MetadataWrapper(module, unsafe_skip_copy=True)
        self.violations = []
        self.visitors = [
            TorchDeprecatedSymbolsVisitor(DEPRECATED_CONFIG_PATH),
            TorchSynchronizedDataLoaderVisitor(),
        ]

    def run(self):
        self.module.visit_batched(self.visitors)
        for v in self.visitors:
            self.violations += v.violations
        for violation in self.violations:
            yield violation.flake8_result()


class TorchCodemod(codemod.Codemod):
    def transform_module_impl(self, module: cst.Module) -> cst.Module:
        # We use `unsafe_skip_copy`` here not only to save some time, but
        # because `deep_replace`` is identity-based and will not work on
        # the original module if the wrapper does a deep copy:
        # in that case we would need to use `wrapped_module.module`
        # instead of `module`.
        wrapped_module = cst.MetadataWrapper(module, unsafe_skip_copy=True)

        violations = []
        visitors = [
            TorchDeprecatedSymbolsVisitor(DEPRECATED_CONFIG_PATH),
            TorchSynchronizedDataLoaderVisitor(),
        ]
        wrapped_module.visit_batched(visitors)
        for v in visitors:
            violations += v.violations

        fixes_count = 0
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

        new_module = deep_multi_replace(module, replacement_map)

        update_imports_visitor = _UpdateFunctorchImports()
        new_module = new_module.visit(update_imports_visitor)

        if fixes_count == 0 and not update_imports_visitor.changed:
            raise codemod.SkipFile("No changes")

        return new_module
