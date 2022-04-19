load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cc_library(
    name = "squareAdd",
    srcs = ["squareAdd.cc"],
    hdrs = ["squareAdd.h"],
    copts = [
        "-x", "cuda",
        "--device-c",
    ],
    linkstatic = 1,
    linkopts = [
        "-x", "cuda",
    ],
)

cuda_library(
    name = "vectorSquareAdd",
    srcs = ["vectorSquareAdd.cu"],
    hdrs = ["vectorSquareAdd.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":squareAdd",
        "@local_config_cuda//cuda:cuda",
    ],
    copts = [
        "--device-c",
    ],
)

cc_binary(
  name='main',
  srcs=['main.cc'],
  deps=[':vectorSquareAdd'],
)
