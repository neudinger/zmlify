bazel run //zerkerus

bazel test //zerkerus:zerkerus_test
bazel test //zerkerus:zerkerus_utils_test
bazel test //zerkerus:zerkerus_math_test
bazel test //zerkerus:zerkerus_flatbuf_test

bazel build //zerkerus:zerkerus_wasm

bazel run //zerkerus:test_iree
