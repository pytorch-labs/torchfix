from pathlib import Path
from torchfix.torchfix import (
    TorchChecker,
    TorchCodemod,
    TorchCodemodConfig,
    GET_ALL_VISITORS,
)
import logging
import libcst.codemod as codemod

FIXTURES_PATH = Path(__file__).absolute().parent / "fixtures"
LOGGER = logging.getLogger(__name__)


def _checker_results(s):
    checker = TorchChecker(None, s)
    return [f"{line}:{col} {msg}" for line, col, msg, _ in checker.run()]


def _codemod_results(source_path):
    with open(source_path) as source:
        code = source.read()
    config = TorchCodemodConfig(select="ALL")
    context = TorchCodemod(codemod.CodemodContext(filename=source_path), config)
    new_module = codemod.transform_module(context, code)
    return new_module.code


def test_empty():
    assert _checker_results([""]) == []


def test_checker_fixtures():
    for source_path in FIXTURES_PATH.glob("**/checker/*.py"):
        LOGGER.info("Testing %s", source_path.relative_to(Path.cwd()))
        expected_path = str(source_path)[:-2] + "txt"
        expected_results = []
        with open(expected_path) as expected:
            for line in expected:
                expected_results.append(line.rstrip())

        with open(source_path) as source:
            assert _checker_results(source.readlines()) == expected_results


def test_codemod_fixtures():
    for source_path in FIXTURES_PATH.glob("**/codemod/*.py"):
        LOGGER.info("Testing %s", source_path.relative_to(Path.cwd()))
        expected_path = source_path.with_suffix(".py.out")
        expected_results = expected_path.read_text()
        assert _codemod_results(source_path) == expected_results


def test_errorcodes_distinct():
    visitors = GET_ALL_VISITORS()
    seen = set()
    for visitor in visitors:
        LOGGER.info("Checking error code for %s", visitor.__class__.__name__)
        error_code = visitor.ERROR_CODE
        for e in error_code if isinstance(error_code, list) else [error_code]:
            assert e not in seen
            seen.add(e)
