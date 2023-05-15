# distutils: language = c
# distutils: sources = ptu_decoder/stream_decoder.c

from cpython.mem cimport PyMem_Free

cdef extern from "stream_decoder.h":
    ctypedef struct CStreamDecoder:
        pass
    CStreamDecoder* StreamDecoder_new()
    void StreamDecoder_delete(CStreamDecoder* sd)
    void StreamDecoder_decode(CStreamDecoder* sd, long long record)
    int StreamDecoder_get_num_records(CStreamDecoder* sd)
    double StreamDecoder_get_most_recent_time(CStreamDecoder* sd)
    long long* StreamDecoder_get_records(CStreamDecoder* sd);

cdef class StreamDecoder:  # now the Cython class can be named StreamDecoder
    cdef CStreamDecoder* c_stream_decoder  # use CStreamDecoder here

    def __cinit__(self):
        self.c_stream_decoder = StreamDecoder_new()

    def __dealloc__(self):
        PyMem_Free(self.c_stream_decoder)

    def decode(self, record):
        StreamDecoder_decode(self.c_stream_decoder, record)

    @property
    def num_records(self):
        return StreamDecoder_get_num_records(self.c_stream_decoder)

    @property
    def records(self):
        cdef int size = StreamDecoder_get_num_records(self.c_stream_decoder)
        cdef long long* c_records = StreamDecoder_get_records(self.c_stream_decoder)
        return [c_records[i] for i in range(size)]  # convert C array to Python list
