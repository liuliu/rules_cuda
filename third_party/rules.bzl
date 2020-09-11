# -*- Python -*-


# Given a source file, generate a test name.
# i.e. "common_runtime/direct_session_test.cc" becomes
#      "common_runtime_direct_session_test"
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_library",
    "if_cuda",
)
# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
  return str(Label(dep))

# LINT.IfChange
def tf_copts():
  return ([
      "-DEIGEN_AVOID_STL_ARRAY",
      "-Iexternal/gemmlowp",
      "-Wno-sign-compare",
      "-fno-exceptions",
  ] + if_cuda(["-DGOOGLE_CUDA=1"]))

def _cuda_copts(opts = []):
    """Gets the appropriate set of copts for (maybe) CUDA compilation.

        If we're doing CUDA compilation, returns copts for our particular CUDA
        compiler.  If we're not doing CUDA compilation, returns an empty list.

        """
    return select({
        "//conditions:default": [],
        "@local_config_cuda//cuda:using_nvcc": ([
            "-nvcc_options=relaxed-constexpr",
            "-nvcc_options=ftz=true",
        ]),
        "@local_config_cuda//cuda:using_clang": ([
            "-fcuda-flush-denormals-to-zero",
        ]),
    })

# Build defs for TensorFlow kernels

# When this target is built using --config=cuda, a cc_library is built
# that passes -DGOOGLE_CUDA=1 and '-x cuda', linking in additional
# libraries needed by GPU kernels.
#
# When this target is built using --config=rocm, a cc_library is built
# that passes -DTENSORFLOW_USE_ROCM and '-x rocm', linking in additional
# libraries needed by GPU kernels.
def tf_gpu_kernel_library(
        srcs,
        copts = [],
        cuda_copts = [],
        deps = [],
        hdrs = [],
        **kwargs):
    copts = copts + tf_copts() + _cuda_copts(opts = cuda_copts)
    kwargs["features"] = kwargs.get("features", []) + ["-use_header_modules"]

    cuda_library(
        srcs = srcs,
        hdrs = hdrs,
        copts = copts,
        deps = deps,
        alwayslink = 1,
        **kwargs
    )

def tf_gpu_library(deps = None, cuda_deps = None, copts = tf_copts(), **kwargs):
    """Generate a cc_library with a conditional set of CUDA dependencies.

      When the library is built with --config=cuda:

      - Both deps and cuda_deps are used as dependencies.
      - The cuda runtime is added as a dependency (if necessary).
      - The library additionally passes -DGOOGLE_CUDA=1 to the list of copts.
      - In addition, when the library is also built with TensorRT enabled, it
          additionally passes -DGOOGLE_TENSORRT=1 to the list of copts.

      Args:
      - cuda_deps: BUILD dependencies which will be linked if and only if:
          '--config=cuda' is passed to the bazel command line.
      - deps: dependencies which will always be linked.
      - copts: copts always passed to the cc_library.
      - kwargs: Any other argument to cc_library.
      """
    if not deps:
        deps = []
    if not cuda_deps:
        cuda_deps = []

    kwargs["features"] = kwargs.get("features", []) + ["-use_header_modules"]
    native.cc_library(
        deps = deps,
        copts = (copts + if_cuda(["-DGOOGLE_CUDA=1"])),
        **kwargs
    )

# terminology changes: saving tf_cuda_* definition for compatibility
def tf_cuda_library(*args, **kwargs):
    tf_gpu_library(*args, **kwargs)

