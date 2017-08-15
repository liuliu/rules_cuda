load("//third_party:rules.bzl", "tf_cuda_library")
load("//third_party:rules.bzl", "tf_gpu_kernel_library")

tf_gpu_kernel_library(
    name = "vectorAdd",
    srcs = ["vectorAdd.cu.cc"],
    hdrs = ["vectorAdd.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda",
        # "//tensorflow/core:gpu_init",
        # "//tensorflow/core:lib",
        # "//tensorflow/core:lib_internal",
        # "//tensorflow/core:stream_executor",
    ],
)

cc_binary(
  name='main',
  srcs=['main.cc'],
  deps=[':vectorAdd'],
)

cc_library(
    name = "cuda",
    data = [
        "@local_config_cuda//cuda:cudart",
    ],
    linkopts = select({
        "@local_config_cuda//cuda:darwin": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib",
        ],
        "//conditions:default": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib64",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib64",
        ],
    }),
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)
