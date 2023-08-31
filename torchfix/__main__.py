import argparse
import libcst.codemod as codemod

import contextlib
import sys
import io
import os

from .torchfix import TorchCodemod, TorchCodemodConfig
from .common import CYAN, ENDC


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
    parser.add_argument(
        "--select",
        help="ALL to enable rules disabled by default",
        choices=[
            "ALL",
        ],
    )

    # XXX TODO: Get rid of this!
    # Silence "Failed to determine module name"
    # https://github.com/Instagram/LibCST/issues/944
    parser.add_argument(
        "--show-stderr",
        action="store_true",
    )

    args = parser.parse_args()

    files = codemod.gather_files(args.path)

    # Filter out files that don't have "torch" string in them.
    # This avoids expensive parsing.
    MARKER = "torch"  # this will catch import torch or functorch
    torch_files = []
    for file in files:
        # TODO: remove the check when https://github.com/Instagram/LibCST/pull/994 lands
        if os.path.isfile(file):  # `codemod.gather_files` can return dirs with ".py"
            with open(file, errors="replace") as f:
                for line in f:
                    if MARKER in line:
                        torch_files.append(file)
                        break

    config = TorchCodemodConfig()
    config.select = args.select
    command_instance = TorchCodemod(codemod.CodemodContext(), config)
    DIFF_CONTEXT = 5
    try:
        if not args.show_stderr:
            context = contextlib.redirect_stderr(io.StringIO())
        else:
            # Should get rid of this code eventually.
            context = contextlib.nullcontext()  # type: ignore
        with context:
            result = codemod.parallel_exec_transform_with_prettyprint(
                command_instance,
                torch_files,
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
