const std = @import("std");
const iree = @import("c");

pub fn main() !void {
    const allocator = iree.iree_allocator_system();

    var instance: ?*iree.iree_hal_driver_registry_t = null;
    const err = iree.iree_hal_driver_registry_allocate(allocator, &instance);

    if (err == null) {
        std.debug.print("Successfully allocated IREE driver registry!\n", .{});
    } else {
        std.debug.print("Failed to allocate IREE driver registry.\n", .{});
        return error.IreeInitFailed;
    }

    iree.iree_hal_driver_registry_free(instance);
    std.debug.print("IREE shutdown successful.\n", .{});
}
