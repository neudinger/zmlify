bazel build //:completion
bazel build //simple
bazel run //simple
bazel test //simple:simple_test

bazel build //resnet-18

# --run_under="lldb --batch -o run -k bt -k exit --"

bazel run //resnet-18 -- --model=$HOME/models/resnet-18 --image=$HOME/dataset/cats-image/cats_image.jpeg
bazel run //zk-resnet -- --model=$HOME/models/resnet-18 --image=$HOME/dataset/cats-image/cats_image.jpeg

bazel test //zk-resnet:fiat_shamir_test
bazel test //zk-resnet:ntt_test

bazel run //zk-resnet:labrador_test_bin

