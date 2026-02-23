bazel build //:completion
bazel build //simple
bazel run //simple
bazel test //simple:simple_test
