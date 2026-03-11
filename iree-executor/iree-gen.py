# pip install iree-compiler
# pip install iree-runtime numpy

import iree.compiler as ireec
import iree.runtime as ireert
import numpy as np

# 1. Define your MLIR source code
# This is a simple module with a function that adds two 4-element f32 tensors.
mlir_source = """
module @math {
  func.func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""

print("Compiling MLIR...")

# 2. Compile the MLIR string into a FlatBuffer
# 'target_backends' specifies what hardware you are compiling for. 
# "llvm-cpu" is standard for native CPU execution. You could also use "vmvx" (reference CPU),
# "vulkan-spirv" (GPU), or "rocm" / "cuda".
compiled_flatbuffer = ireec.compile_str(
    mlir_source,
    target_backends=["llvm-cpu"],
    extra_args=["--iree-llvmcpu-target-cpu=generic", "--iree-consteval-jit-target-device=vmvx"],
)

# 3. Write the compiled bytes to a .vmfb file
output_file = "module.vmfb"
with open(output_file, "wb") as f:
    f.write(compiled_flatbuffer)

print(f"Successfully compiled and saved to {output_file}")

print("Compiling MLIR for Apple Metal...")

# Target the Apple Metal backend
compiled_flatbuffer = ireec.compile_str(
    mlir_source,
    target_backends=["metal-spirv"] 
)

output_file_metal = "module_metal.vmfb"
with open(output_file_metal, "wb") as f:
    f.write(compiled_flatbuffer)

print(f"Successfully compiled and saved to {output_file_metal}")


print("Compiling MLIR for WebAssembly...")

# Target the llvm-cpu backend, but force it to output WebAssembly
compiled_flatbuffer = ireec.compile_str(
    mlir_source,
    target_backends=["llvm-cpu"],
    extra_args=[
        "--iree-llvmcpu-target-triple=wasm32-unknown-emscripten",
        "--iree-llvmcpu-target-cpu=generic"
    ] 
)

output_file_wasm = "module_wasm.vmfb"
with open(output_file_wasm, "wb") as f:
    f.write(compiled_flatbuffer)

print(f"Successfully compiled and saved to {output_file_wasm}")


vmfb_file = "module.vmfb"
print("Loading module into IREE runtime...")

config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)

# FIX: Use mmap directly on the file path instead of reading bytes manually
vm_module = ireert.VmModule.mmap(ctx.instance, vmfb_file)
ctx.add_vm_module(vm_module)

tensor_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
tensor_b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

print(f"Input A: {tensor_a}")
print(f"Input B: {tensor_b}")

result = ctx.modules.math.add(tensor_a, tensor_b)
output_array = np.asarray(result)

print(f"Result : {output_array}")
