#ifndef STREAM_DECODER_H
#define STREAM_DECODER_H

typedef struct CStreamDecoder {
    int num_records;
    double true_time;
    int capacity;
    double overflow_correction;
    long long* records;
} CStreamDecoder;

CStreamDecoder* StreamDecoder_new();
// void StreamDecoder_delete(CStreamDecoder* sd);
void StreamDecoder_decode(CStreamDecoder* sd, long long record);
int StreamDecoder_get_num_records(CStreamDecoder* sd);
long long* StreamDecoder_get_records(CStreamDecoder* sd);

#endif
