#include "zerkerus_fbs_bridge.h"
#include "challenge_json_printer.h"
#include "commitment_json_printer.h"
#include "flatcc/flatcc_json_printer.h"
#include "registration_json_printer.h"
#include "response_json_printer.h"

#include "challenge_json_printer.h"
#include "commitment_json_printer.h"
#include "flatcc/flatcc_json_printer.h"
#include "registration_json_printer.h"
#include "response_json_printer.h"

void print_registration_json_stdout(const void *buf, size_t bufsiz) {
  registration_print_json(0, buf, bufsiz);
}
void print_commitment_json_stdout(const void *buf, size_t bufsiz) {
  commitment_print_json(0, buf, bufsiz);
}
void print_challenge_json_stdout(const void *buf, size_t bufsiz) {
  challenge_print_json(0, buf, bufsiz);
}
void print_response_json_stdout(const void *buf, size_t bufsiz) {
  response_print_json(0, buf, bufsiz);
}
