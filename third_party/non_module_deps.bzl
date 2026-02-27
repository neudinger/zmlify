load("//third_party/stb:repo.bzl", stb = "repo")
load("//third_party/flatbuf:repo.bzl", flatbuf = "repo")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _non_module_deps_impl(mctx):
    stb()
    flatbuf()

    # IREE requires the root module to provide llvm-raw mapping to LLVM Project
    # Commit used by IREE v3.10.0 submodule: 12d80f913c5a0f13f54610e9b0524137301665e3
    if not bool(native.existing_rule("llvm-raw")):
        http_archive(
            name = "llvm-raw",
            build_file_content = "# empty",
            url = "https://github.com/llvm/llvm-project/archive/12d80f913c5a0f13f54610e9b0524137301665e3.tar.gz",
            strip_prefix = "llvm-project-12d80f913c5a0f13f54610e9b0524137301665e3",
        )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

non_module_deps = module_extension(
    implementation = _non_module_deps_impl,
)
