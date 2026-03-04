bazel test //zk-crypto:fiat_shamir_test && \
bazel run //zk-crypto:ntt_test_bin && \
bazel run //zk-crypto:labrador_test_bin && \
bazel build //zk-crypto:zk-crypto-wasm
