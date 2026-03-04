#include "iree/vm/api.h"

// Helper to create an input list and push two i32 values
iree_status_t shim_create_i32_inputs(int32_t a, int32_t b,
                                     iree_allocator_t allocator,
                                     iree_vm_list_t **out_list) {
  iree_vm_list_t *list = NULL;

  // Pass a zero-initialized type def to create a variant list
  iree_status_t status =
      iree_vm_list_create((iree_vm_type_def_t){0}, 2, allocator, &list);
  if (!iree_status_is_ok(status))
    return status;

  iree_vm_value_t val_a = iree_vm_value_make_i32(a);
  status = iree_vm_list_push_value(list, &val_a);
  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(list);
    return status;
  }

  iree_vm_value_t val_b = iree_vm_value_make_i32(b);
  status = iree_vm_list_push_value(list, &val_b);
  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(list);
    return status;
  }

  *out_list = list;
  return iree_ok_status();
}

// Helper to create an empty list to receive outputs
iree_status_t shim_create_empty_list(iree_host_size_t capacity,
                                     iree_allocator_t allocator,
                                     iree_vm_list_t **out_list) {
  // Pass a zero-initialized type def to create a variant list
  return iree_vm_list_create((iree_vm_type_def_t){0}, capacity, allocator,
                             out_list);
}

// Helper to extract an i32 from an output list
iree_status_t shim_get_i32_output(iree_vm_list_t *list, iree_host_size_t index,
                                  int32_t *out_val) {
  iree_vm_value_t val;
  iree_status_t status = iree_vm_list_get_value(list, index, &val);
  if (!iree_status_is_ok(status))
    return status;

  *out_val = val.i32;
  return iree_ok_status();
}
