def check_schema(name, schema, include_paths = []):
    """Generates a command to check the schema using flatcc."""
    
    cmd = "$(location @flatcc//:flatcc)"
    for path in include_paths:
        cmd += " -I %s" % path
    cmd += " -a -o $(@D)/gen_check $(SRCS)"

    native.genrule(
        name = name + "_check_genrule",
        srcs = [schema],
        outs = ["gen_check/dummy.txt"], # Just dummy to enforce run
        cmd = cmd + " && touch $@",
        tools = ["@flatcc//:flatcc"],
        message = "Checking FlatBuffers schema: %s" % schema,
        visibility = ["//visibility:private"],
    )

def flatcc_library(name, src, outs = [], include_paths = [], schema_deps = []):
    """
    Generates C headers from a FlatBuffers schema file using flatcc, 
    and creates a cc_library containing them.
    
    Args:
        name: Name of the cc_library target.
        src: The .fbs schema file.
        outs: The expected generated files (e.g. ["my_schema_builder.h", "my_schema_reader.h"])
        include_paths: Optional additional paths to append to flatcc's include path
        schema_deps: Additional .fbs files needed for include resolution
    """
    cmd = "$(location @flatcc//:flatcc) -a "
    for path in include_paths:
        cmd += "-I %s " % path
    # Auto-include the directory of each schema dep for include resolution
    for dep in schema_deps:
        cmd += "-I $$(dirname $(location %s)) " % dep
    cmd += "--json --json-printer -o $(@D) $(location %s)" % src

    native.genrule(
        name = name + "_gen",
        srcs = [src] + schema_deps,
        outs = outs,
        cmd = cmd,
        tools = ["@flatcc//:flatcc"],
        message = "Generating FlatBuffers C headers for %s" % src,
        visibility = ["//visibility:private"],
    )

    native.cc_library(
        name = name,
        hdrs = outs,
        includes = ["."],
        deps = ["@flatcc//:runtime"],
        visibility = ["//visibility:public"],
    )
