const std = @import("std");
const c_interface = @import("c");

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const zig_allocator = gpa.allocator();

    const args = init.minimal.args;

    const vmfb_path = if (args.vector.len > 1) std.mem.span(args.vector[1]) else "simple_cpu.vmfb";

    if (args.vector.len <= 1) {
        std.debug.print("Usage: test_iree <vmfb_path>\n", .{});
        std.debug.print("Defaulting to 'simple_cpu.vmfb'\n", .{});
    }

    const allocator = c_interface.iree_allocator_system();

    // 1. Create IREE instance
    var instance: ?*c_interface.iree_vm_instance_t = null;
    try checkStatus(c_interface.iree_vm_instance_create(0, allocator, &instance), "iree_vm_instance_create");
    defer c_interface.iree_vm_instance_release(instance);

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

    // 7. Lookup and invoke function
    // We'll try to find the first exported function if 'main' isn't found
    const module_ptr = bytecode_module.?;
    const signature = c_interface.iree_vm_module_signature(module_ptr);

    if (signature.export_function_count == 0) {
        return error.NoExportedFunctions;
    }

    var function: c_interface.iree_vm_function_t = undefined;
    try checkStatus(c_interface.iree_vm_module_lookup_function_by_name(module_ptr, c_interface.IREE_VM_FUNCTION_LINKAGE_EXPORT, c_interface.iree_make_cstring_view("main"), &function), "Lookup main");

    std.debug.print("Executing function 'main'...\n", .{});

    try checkStatus(c_interface.iree_vm_invoke(context, function, c_interface.IREE_VM_INVOCATION_FLAG_NONE, null, // policy
        null, // inputs
        null, // outputs
        allocator), "iree_vm_invoke");

    std.debug.print("Execution successful!\n", .{});
}

fn checkStatus(status: c_interface.iree_status_t, context_msg: []const u8) !void {
    if (status) |s| {
        defer _ = c_interface.iree_status_ignore(s);
        std.debug.print("IREE Error at {s}\n", .{context_msg});
        return error.IreeError;
    }
}
