#include "all_builder.h"
#include "all_reader.h"
#include "flatbuffers_common_builder.h"
#include "flatbuffers_common_reader.h"
#include <stddef.h>
#include <stdint.h>

void print_registration_json_stdout(const void *buf, size_t bufsiz);
void print_commitment_json_stdout(const void *buf, size_t bufsiz);
void print_challenge_json_stdout(const void *buf, size_t bufsiz);
void print_response_json_stdout(const void *buf, size_t bufsiz);
