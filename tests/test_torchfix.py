from pathlib import Path
from torchfix.torchfix import (
    TorchChecker,
    TorchCodemod,
    TorchCodemodConfig,
    DISABLED_BY_DEFAULT,
    expand_error_codes,
    GET_ALL_VISITORS,
    GET_ALL_ERROR_CODES,
    process_error_code_str,
)
import logging
import libcst.codemod as codemod

FIXTURES_PATH = Path(__file__).absolute().parent / "fixtures"
LOGGER = logging.getLogger(__name__)


def pytest_generate_tests(metafunc):
    # Dynamically generate test cases from paths
    if "checker_source_path" in metafunc.fixturenames:
        files = list(FIXTURES_PATH.glob("**/checker/*.py"))
        metafunc.parametrize(
            "checker_source_path", files, ids=[file_name.stem for file_name in files]
        )
    if "codemod_source_path" in metafunc.fixturenames:
        files = list(FIXTURES_PATH.glob("**/codemod/*.in.py"))
        metafunc.parametrize(
            "codemod_source_path", files, ids=[file_name.stem for file_name in files]
        )
    if "case" in metafunc.fixturenames:
        exclude_set = expand_error_codes(tuple(DISABLED_BY_DEFAULT))
        cases = [
            ("ALL", GET_ALL_ERROR_CODES()),
            ("ALL,TOR102", GET_ALL_ERROR_CODES()),
            ("TOR102", {"TOR102"}),
            ("TOR102,TOR101", {"TOR102", "TOR101"}),
            (
                "TOR1,TOR102",
                {
                    "TOR101",
                    "TOR102",
                    "TOR103",
                    "TOR104",
                    "TOR105",
                    "TOR106",
                    "TOR107",
                },
            ),
            (None, set(GET_ALL_ERROR_CODES()) - exclude_set),
        ]
        metafunc.parametrize("case,expected", cases, ids=[case for case, _ in cases])


def _checker_results(s):
    checker = TorchChecker(None, s)
    return [f"{line}:{col} {msg}" for line, col, msg, _ in checker.run()]


def _codemod_results(source_path: Path):
    code = source_path.read_text()
    config = TorchCodemodConfig(select=list(GET_ALL_ERROR_CODES()))
    context = TorchCodemod(codemod.CodemodContext(filename=str(source_path)), config)
    new_module = codemod.transform_module(context, code)
    if isinstance(new_module, codemod.TransformSuccess):
        return new_module.code
    if isinstance(new_module, codemod.TransformFailure):
        raise new_module.error


def test_empty():
    assert _checker_results([""]) == []


def test_checker_fixtures(checker_source_path: Path):
    expected_path = checker_source_path.with_suffix(".txt")
    expected_results = expected_path.read_text().splitlines()
    results = _checker_results(
        checker_source_path.read_text().splitlines(keepends=True)
    )
    # Overwrite the expected data with the results (useful when updating tests)
    # expected_path.write_text("".join([f"{line}\n" for line in results]))
    assert results == expected_results


def test_codemod_fixtures(codemod_source_path: Path):
    expected_path = codemod_source_path.with_stem(
        codemod_source_path.stem.replace(".in", ".out")
    )
    expected_results = expected_path.read_text()
    assert _codemod_results(codemod_source_path) == expected_results


def test_errorcodes_distinct():
    visitors = GET_ALL_VISITORS()
    seen = set()
    for visitor in visitors:
        LOGGER.info("Checking error code for %s", visitor.__class__.__name__)
        for e in visitor.ERRORS:
            assert e.error_code not in seen
            seen.add(e.error_code)


def test_parse_error_code_str(case, expected):
    assert process_error_code_str(case) == expected
