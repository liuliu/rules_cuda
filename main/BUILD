load("//third_party:rules.bzl", "tf_cuda_library")
load("//third_party:rules.bzl", "tf_gpu_kernel_library")

tf_gpu_kernel_library(
    name = "vectorAdd",
    srcs = ["vectorAdd.cu.cc"],
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
