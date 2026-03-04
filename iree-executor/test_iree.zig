const std = @import("std");
const c_interface = @import("c");

// ADD THESE THREE LINES:
extern fn shim_create_i32_inputs(a: i32, b: i32, allocator: c_interface.iree_allocator_t, out_list: *?*c_interface.iree_vm_list_t) c_interface.iree_status_t;
extern fn shim_create_empty_list(capacity: usize, allocator: c_interface.iree_allocator_t, out_list: *?*c_interface.iree_vm_list_t) c_interface.iree_status_t;
extern fn shim_get_i32_output(list: *c_interface.iree_vm_list_t, index: usize, out_val: *i32) c_interface.iree_status_t;

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const zig_allocator = gpa.allocator();

    const args = init.minimal.args;

    const vmfb_path = if (args.vector.len > 1) std.mem.span(args.vector[1]) else "simple_cpu.vmfb";

    if (args.vector.len <= 1) {
        std.debug.print("Usage: test_iree <vmfb_path>\n", .{});
        std.debug.print("Defaulting to 'simple_cpu.vmfb'\n", .{});
        return;
    }

    const allocator = c_interface.iree_allocator_system();

    // 1. Create IREE instance
    var instance: ?*c_interface.iree_vm_instance_t = null;
    try checkStatus(c_interface.iree_vm_instance_create(c_interface.IREE_VM_TYPE_CAPACITY_DEFAULT, allocator, &instance), "iree_vm_instance_create");
    defer c_interface.iree_vm_instance_release(instance);
    try checkStatus(c_interface.iree_hal_module_register_all_types(instance), "iree_hal_module_register_all_types");

    // 2. Register HAL drivers and create device
    try checkStatus(c_interface.iree_hal_local_task_driver_module_register(c_interface.iree_hal_driver_registry_default()), "iree_hal_local_task_driver_module_register");

    var device: ?*c_interface.iree_hal_device_t = null;
    try checkStatus(c_interface.iree_hal_create_device(c_interface.iree_hal_driver_registry_default(), c_interface.iree_make_cstring_view("local-task"), allocator, &device), "iree_hal_create_device");
    defer c_interface.iree_hal_device_release(device);

    // 3. Load VMFB file
    const dir = std.Io.Dir.cwd();
    var file = try dir.openFile(init.io, vmfb_path, .{});
    defer file.close(init.io);
    const file_stat = try file.stat(init.io);
    const vmfb_data = try zig_allocator.alloc(u8, file_stat.size);
    defer zig_allocator.free(vmfb_data);
    _ = try file.readPositionalAll(init.io, vmfb_data, 0);

    // 4. Create bytecode module
    var bytecode_module: ?*c_interface.iree_vm_module_t = null;
    try checkStatus(c_interface.iree_vm_bytecode_module_create(instance, c_interface.IREE_VM_BYTECODE_MODULE_FLAG_NONE, c_interface.iree_make_const_byte_span(vmfb_data.ptr, vmfb_data.len), c_interface.iree_allocator_null(), allocator, &bytecode_module), "iree_vm_bytecode_module_create");
    defer c_interface.iree_vm_module_release(bytecode_module);

    // 5. Create HAL module
    var hal_module: ?*c_interface.iree_vm_module_t = null;
    try checkStatus(c_interface.iree_hal_module_create(instance, c_interface.iree_hal_module_device_policy_default(), 1, &device, c_interface.IREE_HAL_MODULE_FLAG_NONE, c_interface.iree_hal_module_debug_sink_null(), allocator, &hal_module), "iree_hal_module_create");
    defer c_interface.iree_vm_module_release(hal_module);

    // 6. Create context
    var context: ?*c_interface.iree_vm_context_t = null;
    var modules = [_]?*c_interface.iree_vm_module_t{ hal_module, bytecode_module };
    try checkStatus(c_interface.iree_vm_context_create_with_modules(instance, c_interface.IREE_VM_CONTEXT_FLAG_NONE, modules.len, &modules, allocator, &context), "iree_vm_context_create_with_modules");
    defer c_interface.iree_vm_context_release(context);

    // 7. Lookup the first exported function dynamically
    const module_ptr = bytecode_module.?;
    const signature = c_interface.iree_vm_module_signature(module_ptr);

    if (signature.export_function_count == 0) {
        return error.NoExportedFunctions;
    }

    var function: c_interface.iree_vm_function_t = undefined;

    // Call with 4 arguments (removed &export_name)
    try checkStatus(c_interface.iree_vm_module_lookup_function_by_ordinal(module_ptr, c_interface.IREE_VM_FUNCTION_LINKAGE_EXPORT, 0, // Get the first exported function (ordinal 0)
        &function), "Lookup first exported function");

    // Fetch the name using the dedicated API
    const export_name = c_interface.iree_vm_function_name(&function);
    const name_slice = export_name.data[0..export_name.size];
    std.debug.print("Executing function '{s}'...\n", .{name_slice});

    // 8. Prepare Inputs (10 and 32)
    var inputs: ?*c_interface.iree_vm_list_t = null;
    try checkStatus(shim_create_i32_inputs(10, 32, allocator, &inputs), "shim_create_i32_inputs");
    defer c_interface.iree_vm_list_release(inputs);

    // 9. Prepare Outputs (expecting 1 result)
    var outputs: ?*c_interface.iree_vm_list_t = null;
    try checkStatus(shim_create_empty_list(1, allocator, &outputs), "shim_create_empty_list");
    defer c_interface.iree_vm_list_release(outputs);

    // 10. Invoke the function
    try checkStatus(c_interface.iree_vm_invoke(context, function, c_interface.IREE_VM_INVOCATION_FLAG_NONE, null, // policy
        inputs, outputs, allocator), "iree_vm_invoke");

    // 11. Read and print the result
    var result: i32 = 0;
    try checkStatus(shim_get_i32_output(outputs.?, 0, &result), "shim_get_i32_output");

    std.debug.print("Execution successful! 10 + 32 = {d}\n", .{result});
}

fn checkStatus(status: c_interface.iree_status_t, context_msg: []const u8) !void {
    if (status) |s| {
        defer _ = c_interface.iree_status_ignore(s);
        std.debug.print("IREE Error at {s}\n", .{context_msg});
        return error.IreeError;
    }
}
