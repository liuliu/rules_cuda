load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cuda_library(
    name = "vectorAdd",
    srcs = ["vectorAdd.cu"],
    hdrs = ["vectorAdd.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda",
    ],
)

cc_binary(
  name='main',
  srcs=['main.cc'],
  deps=[':vectorAdd'],
)
