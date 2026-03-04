// ./iree-build-host/install/bin/iree-compile \
//   --iree-hal-target-backends=llvm-cpu \
//   example.mlir \
//   -o example_mac.vmfb

// ./iree-build-host/install/bin/iree-run-module \
//   --module=example_mac.vmfb \      
//   --device=local-sync \
//   --function=add \
//   --input="i32=10" \
//   --input="i32=32"

module @arith_example {
  func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %res = arith.addi %arg0, %arg1 : i32
    return %res : i32
  }
}
