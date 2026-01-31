"""CUDA extension loader for ch12 kernels."""

from pathlib import Path

from core.utils.extension_loader_template import load_cuda_extension_v2
from core.profile.nvtx_stub import ensure_nvtx_stub

NVTX_CFLAG = "-DENABLE_NVTX_PROFILING"
NVCC_ALLOW_UNSUPPORTED = "--allow-unsupported-compiler"
_NVTX_STUB_LIB = ensure_nvtx_stub()
NVTX_LDFLAGS = [f"-L{_NVTX_STUB_LIB.parent}", "-lnvToolsExt"]

_EXTENSION_DIR = Path(__file__).parent
_COMMON_HEADERS = _EXTENSION_DIR.parent.parent / "core" / "common" / "headers"


def _cuda_flags() -> list[str]:
    return ["-lineinfo", f"-I{_COMMON_HEADERS}", NVTX_CFLAG, NVCC_ALLOW_UNSUPPORTED]


def load_kernel_fusion_extension():
    """Load the kernel fusion CUDA extension."""
    return load_cuda_extension_v2(
        name="kernel_fusion_kernels",
        sources=[_EXTENSION_DIR / "kernel_fusion_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=list(NVTX_LDFLAGS),
    )


def load_graph_bandwidth_extension():
    """Load the graph bandwidth CUDA extension."""
    return load_cuda_extension_v2(
        name="graph_bandwidth_kernels",
        sources=[_EXTENSION_DIR / "graph_bandwidth_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=list(NVTX_LDFLAGS),
    )


def load_work_queue_extension():
    """Load the work queue CUDA extension."""
    return load_cuda_extension_v2(
        name="work_queue_kernels",
        sources=[_EXTENSION_DIR / "work_queue_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=list(NVTX_LDFLAGS),
    )


def load_cuda_graphs_extension():
    """Load the CUDA graphs extension."""
    return load_cuda_extension_v2(
        name="cuda_graphs_kernels",
        sources=[_EXTENSION_DIR / "cuda_graphs_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=list(NVTX_LDFLAGS),
    )




"""CUDA extension loader for ch06 kernels."""

from pathlib import Path

from core.utils.extension_loader_template import load_cuda_extension_v2
from core.profile.nvtx_stub import ensure_nvtx_stub

NVTX_CFLAG = "-DENABLE_NVTX_PROFILING"
_NVTX_STUB_LIB = ensure_nvtx_stub()
NVTX_LDFLAGS = [f"-L{_NVTX_STUB_LIB.parent}", "-lnvToolsExt"]

_EXTENSION_DIR = Path(__file__).parent
_COMMON_HEADERS = _EXTENSION_DIR.parent.parent / "core" / "common" / "headers"


def _cuda_flags() -> list[str]:
    return ["-lineinfo", f"-I{_COMMON_HEADERS}", NVTX_CFLAG]


def load_coalescing_extension():
    """Load the coalescing kernels CUDA extension."""
    return load_cuda_extension_v2(
        name="coalescing_kernels",
        sources=[_EXTENSION_DIR / "coalescing_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=list(NVTX_LDFLAGS),
    )


def load_bank_conflicts_extension():
    """Load the bank conflicts kernels CUDA extension."""
    return load_cuda_extension_v2(
        name="bank_conflicts_kernels",
        sources=[_EXTENSION_DIR / "bank_conflicts_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=list(NVTX_LDFLAGS),
    )


def load_ilp_extension():
    """Load the ILP kernels CUDA extension."""
    return load_cuda_extension_v2(
        name="ilp_kernels",
        sources=[_EXTENSION_DIR / "ilp_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=list(NVTX_LDFLAGS),
    )


def load_launch_bounds_extension():
    """Load the launch bounds CUDA extension."""
    return load_cuda_extension_v2(
        name="launch_bounds_kernels",
        sources=[_EXTENSION_DIR / "launch_bounds_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=list(NVTX_LDFLAGS),
    )