load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "compiler_lib",
    srcs = [
        "external/hash/cmetrohash64.c",
        "external/hash/str_set.c",
        "external/hash/ptr_set.c",
        "src/compiler/hash_tables/symbol_table.c",
        "src/compiler/hash_tables/scope_table.c",
        "src/compiler/hash_tables/name_table.c",
        "src/compiler/hash_tables/schema_table.c",
        "src/compiler/hash_tables/value_set.c",
        "src/compiler/fileio.c",
        "src/compiler/parser.c",
        "src/compiler/semantics.c",
        "src/compiler/coerce.c",
        "src/compiler/flatcc.c",
        "src/compiler/codegen_c.c",
        "src/compiler/codegen_c_reader.c",
        "src/compiler/codegen_c_sort.c",
        "src/compiler/codegen_c_builder.c",
        "src/compiler/codegen_c_verifier.c",
        "src/compiler/codegen_c_sorter.c",
        "src/compiler/codegen_c_json_parser.c",
        "src/compiler/codegen_c_json_printer.c",
    ],
    hdrs = glob([
        "include/**/*.h",
        "config/**/*.h",
        "external/**/*.h",
        "src/**/*.h",
        "external/lex/luthor.c", # Required by parser.c
    ]),
    copts = [
        "-Wno-unused-function",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-variable",
    ],
    includes = [
        "config",
        "include",
        "external",
    ],
    visibility = ["//visibility:private"],
)

cc_binary(
    name = "flatcc",
    srcs = ["src/cli/flatcc_cli.c"],
    includes = [
        "config",
        "include",
    ],
    deps = [":compiler_lib"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "runtime",
    srcs = [
        "src/runtime/builder.c",
        "src/runtime/emitter.c",
        "src/runtime/refmap.c",
        "src/runtime/verifier.c",
        "src/runtime/json_parser.c",
        "src/runtime/json_printer.c",
    ],
    hdrs = glob([
        "include/**/*.h",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
