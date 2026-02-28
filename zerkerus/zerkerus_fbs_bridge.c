#include "zerkerus_fbs_bridge.h"
#include "challenge_json_printer.h"
#include "commitment_json_printer.h"
#include "flatcc/flatcc_json_printer.h"
#include "registration_json_printer.h"
#include "response_json_printer.h"
#include <stdio.h>

#include "challenge_json_printer.h"
#include "commitment_json_printer.h"
#include "flatcc/flatcc_json_printer.h"
#include "registration_json_printer.h"
#include "response_json_printer.h"

void print_registration_json_stderr(const void *buf, size_t bufsiz) {
  flatcc_json_printer_t ctx;
  flatcc_json_printer_init(&ctx, stderr);
  zerkerus_Registration_print_json_as_root(&ctx, buf, bufsiz, 0);
  flatcc_json_printer_flush(&ctx);
  fflush(stderr);
}
void print_commitment_json_stderr(const void *buf, size_t bufsiz) {
  flatcc_json_printer_t ctx;
  flatcc_json_printer_init(&ctx, stderr);
  zerkerus_Commitment_print_json_as_root(&ctx, buf, bufsiz, 0);
  flatcc_json_printer_flush(&ctx);
  fflush(stderr);
}
void print_challenge_json_stderr(const void *buf, size_t bufsiz) {
  flatcc_json_printer_t ctx;
  flatcc_json_printer_init(&ctx, stderr);
  zerkerus_Challenge_print_json_as_root(&ctx, buf, bufsiz, 0);
  flatcc_json_printer_flush(&ctx);
  fflush(stderr);
}
void print_response_json_stderr(const void *buf, size_t bufsiz) {
  flatcc_json_printer_t ctx;
  flatcc_json_printer_init(&ctx, stderr);
  zerkerus_Response_print_json_as_root(&ctx, buf, bufsiz, 0);
  flatcc_json_printer_flush(&ctx);
  fflush(stderr);
}
