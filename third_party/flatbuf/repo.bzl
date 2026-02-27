load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "flatcc",
        urls = ["https://github.com/dvidelabs/flatcc/archive/refs/tags/v0.6.1.tar.gz"],
        strip_prefix = "flatcc-0.6.1",
        build_file = "//third_party/flatbuf:flatcc.BUILD",
    )
