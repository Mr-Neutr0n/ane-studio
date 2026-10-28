#include <os/log.h>
#include <os/signpost.h>

static os_log_t inference_log;
static os_signpost_id_t inference_id;

void inference_begin(void) {
    inference_log = os_log_create("dev.harikrishna.anestudio", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
    inference_id = os_signpost_id_generate(inference_log);
    os_signpost_interval_begin(inference_log, inference_id, "local_generation");
}

void inference_end(void) {
    os_signpost_interval_end(inference_log, inference_id, "local_generation");
}
