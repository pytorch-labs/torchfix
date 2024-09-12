# TorchFix - a linter for PyTorch-using code with autofix support

[![PyPI](https://img.shields.io/pypi/v/torchfix.svg)](https://pypi.org/project/torchfix/)

TorchFix is a Python code static analysis tool - a linter with autofix capabilities -
for users of PyTorch. It can be used to find and fix issues like usage of deprecated
PyTorch functions and non-public symbols, and to adopt PyTorch best practices in general.

TorchFix is built upon https://github.com/Instagram/LibCST - a library to manipulate
Python concrete syntax trees. LibCST enables "codemods" (autofixes) in addition to
reporting issues.

TorchFix can be used as a Flake8 plugin (linting only) or as a standalone
program (with autofix available for a subset of the lint violations).

> [!WARNING]
> Currently TorchFix is in a **beta version** stage, so there are still a lot of rough
edges and many things can and will change.

## Installation

To install the latest code from GitHub, clone/download
https://github.com/pytorch-labs/torchfix and run `pip install .`
inside the directory.

To install a release version from PyPI, run `pip install torchfix`.

## Usage

After the installation, TorchFix will be available as a Flake8 plugin, so running
Flake8 normally will run the TorchFix linter.

To see only TorchFix warnings without the rest of the Flake8 linters, you can run
`flake8 --isolated --select=TOR0,TOR1,TOR2`

TorchFix can also be run as a standalone program: `torchfix .`
Add `--fix` parameter to try to autofix some of the issues (the files will be overwritten!)
To see some additional debug info, add `--show-stderr` parameter.

> [!CAUTION]
> Please keep in mind that autofix is a best-effort mechanism. Given the dynamic nature of Python,
and especially the beta version status of TorchFix, it's very difficult to have
certainty when making changes to code, even for the seemingly trivial fixes.

Warnings for issues with codes starting with TOR0, TOR1, and TOR2 are enabled by default.
Warnings with other codes may be too noisy, so not enabled by default.
To enable them, use standard flake8 configuration options for the plugin mode or use
`torchfix --select=ALL .` for the standalone mode.


## Reporting problems

If you encounter a bug or some other problem with TorchFix, please file an issue on
https://github.com/pytorch-labs/torchfix/issues.

## Rule Code Assignment Policy

New rule codes are assigned incrementally across the following categories:

* **TOR0XX, TOR1XX**: General-purpose `torch` functionality.
* **TOR2XX**: Domain-specific rules, such as TorchVision.
* **TOR4XX**: Noisy rules that are disabled by default.
* **TOR9XX**: Internal rules specific for `pytorch/pytorch` repo, other users should not use these.

TOR0, TOR1 and TOR2 are enabled by default.

## Rules

### TOR001 Use of removed function

#### torch.solve

This function was deprecated since PyTorch version 1.9 and is now removed.

`torch.solve` is deprecated in favor of `torch.linalg.solve`.
`torch.linalg.solve` has its arguments reversed and does not return the LU factorization.

To get the LU factorization see `torch.lu`, which can be used with `torch.lu_solve` or `torch.lu_unpack`.

`X = torch.solve(B, A).solution` should be replaced with `X = torch.linalg.solve(A, B)`.

#### torch.symeig

This function was deprecated since PyTorch version 1.9 and is now removed.

`torch.symeig` is deprecated in favor of `torch.linalg.eigh`.

The default behavior has changed from using the upper triangular portion of the matrix by default to using the lower triangular portion.

```python
L, _ = torch.symeig(A, upper=upper)
```

should be replaced with

```python
L = torch.linalg.eigvalsh(A, UPLO='U' if upper else 'L')
```

and

```python
L, V = torch.symeig(A, eigenvectors=True)
```

should be replaced with

```python
L, V = torch.linalg.eigh(A, UPLO='U' if upper else 'L')
```

### TOR002 Likely typo `require_grad` in assignment. Did you mean `requires_grad`?

This is a common misspelling that can lead to silent performance issues.

### TOR003 Please pass `use_reentrant` explicitly to `checkpoint`

The default value of the `use_reentrant` parameter in `torch.utils.checkpoint` is being changed
from `True` to `False`. In the meantime, the value needs to be passed explicitly.

See this [forum post](https://dev-discuss.pytorch.org/t/bc-breaking-update-to-torch-utils-checkpoint-not-passing-in-use-reentrant-flag-will-raise-an-error/1745)
for details.

### TOR004 Import of removed function

See `TOR001`.

### TOR101 Use of deprecated function

#### torch.nn.utils.weight_norm

This function is deprecated. Use `torch.nn.utils.parametrizations.weight_norm`
which uses the modern parametrization API. The new `weight_norm` is compatible
with `state_dict` generated from old `weight_norm`.

Migration guide:

* The magnitude (``weight_g``) and direction (``weight_v``) are now expressed
    as ``parametrizations.weight.original0`` and ``parametrizations.weight.original1``
    respectively.

* To remove the weight normalization reparametrization, use
    `torch.nn.utils.parametrize.remove_parametrizations`.

* The weight is no longer recomputed once at module forward; instead, it will
    be recomputed on every access.  To restore the old behavior, use
    `torch.nn.utils.parametrize.cached` before invoking the module
    in question.

#### torch.backends.cuda.sdp_kernel

This function is deprecated. Use the `torch.nn.attention.sdpa_kernel` context manager instead.

Migration guide:
Each boolean input parameter (defaulting to true unless specified) of `sdp_kernel` corresponds to a `SDPBackened`. If the input parameter is true, the corresponding backend should be added to the input list of `sdpa_kernel`.

#### torch.chain_matmul

This function is deprecated in favor of `torch.linalg.multi_dot`.

Migration guide:
`multi_dot` accepts a list of two or more tensors whereas `chain_matmul` accepted multiple tensors as input arguments. For migration, convert the multiple tensors in argument of  `chain_matmul` into a list of two or more tensors for `multi_dot`.

Example: Replace `torch.chain_matmul(a, b, c)` with `torch.linalg.multi_dot([a, b, c])`.

#### torch.cholesky

`torch.cholesky()` is deprecated in favor of `torch.linalg.cholesky()`.

Migration guide:
* `L = torch.cholesky(A)` should be replaced with `L = torch.linalg.cholesky(A)`.
* `L = torch.cholesky(A, upper=True)` should be replaced with `L = torch.linalg.cholesky(A).mH`

#### torch.qr

`torch.qr()` is deprecated in favor of `torch.linalg.qr()`.

Migration guide:
* The usage `Q, R = torch.qr(A)` should be replaced with `Q, R = torch.linalg.qr(A)`.
* The boolean parameter `some` of `torch.qr` is replaced with a string parameter `mode` in `torch.linalg.qr`. The corresponding change in usage is from `Q, R = torch.qr(A, some=False)` to `Q, R = torch.linalg.qr(A, mode="complete")`.

#### torch.range

The function `torch.range()` is deprecated as its usage is incompatible with Python's builtin range. Instead, use `torch.arange()` as it produces values in `[start, end)`.

Migration guide:
* `torch.range(start, end)` produces values in the range of `[start, end]`. But `torch.arange(start, end)` produces values in `[start, end)`. For step size of 1, migrate usage from `torch.range(start, end, 1)` to `torch.arange(start, end+1, 1)`.

### TOR102 `torch.load` without `weights_only` parameter is unsafe.

Explicitly set `weights_only` to False only if you trust the data you load and full pickle functionality is needed, otherwise set `weights_only=True`.

### TOR103 Import of deprecated function

See `TOR101`.

## License

TorchFix is BSD License licensed, as found in the LICENSE file.
