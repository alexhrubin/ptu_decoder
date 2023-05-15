#include <stdlib.h>
#include "stream_decoder.h"

double T2WRAPAROUND = 210698240;
int resolution = 4;  // 4 picoseconds sample

CStreamDecoder* StreamDecoder_new() {
    CStreamDecoder* sd = (CStreamDecoder*)malloc(sizeof(CStreamDecoder));
    sd->num_records = 0;
    sd->overflow_correction = 0;
    sd->true_time = 0;
    sd->capacity = ;
    sd->records = malloc(10000 * sizeof(long long));
    return sd;
}

union {
    unsigned int allbits;
    struct {
        unsigned time   :28;
        unsigned channel  :4;
    } bits;
} Record;

void StreamDecoder_decode(CStreamDecoder* sd, long long record) {
    Record.allbits = record;  // Decode the record into channel and time parts
    if (Record.bits.channel == 0xF) {
        if ((Record.bits.time & 0xF) == 0) {  // lowest 4 bits are marker bits
            sd->overflow_correction += T2WRAPAROUND;
        }
        else {
            sd->true_time += Record.bits.time;
        }
    }
    else {
        // photon event. We need to add the true_time into the records array.
        sd->true_time = sd->overflow_correction + Record.bits.time;
        sd->records[sd->num_records] = sd->true_time * resolution;
        sd->num_records++;
    }
}

int StreamDecoder_get_num_records(CStreamDecoder* sd) {
    return sd->num_records;
}

long long* StreamDecoder_get_records(CStreamDecoder* sd) {
    return sd->records;
}
