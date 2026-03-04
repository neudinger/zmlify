bazel run //zk-resnet -- \
    --model=/Users/kevin/models/resnet-18/model.safetensors \
    --image=/Users/kevin/dataset/cats-image/cats_image.jpeg && \
bazel test //zk-resnet:sampler_test && \
bazel build //zk-resnet:hash_model_bin && \
bazel build //zk-resnet:vk_hash_lib
