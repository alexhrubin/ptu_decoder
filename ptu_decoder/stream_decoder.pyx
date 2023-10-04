from libc.stdio cimport FILE, fopen, fread, fseek, fclose, SEEK_SET, SEEK_END, ftell

TIME_OVERFLOW = 210698240


cdef class StreamDecoder:
    cdef FILE *fp
    cdef long long overflow_correction
    cdef unsigned int num_events
    cdef dict photon_timestamps

    def __cinit__(self, str file_path):
        self.fp = fopen(file_path.encode('utf-8'), "rb")
        if self.fp is NULL:
            raise FileNotFoundError("Could not open file")
        self.overflow_correction = 0
        self.photon_timestamps = {}  # keys are channels, values are timestamps

    def times_between_channels(self):
        timing_data_starts = 3584 + 48;
        fseek(self.fp, 0, SEEK_END);  # move to end of file
        cdef long int eof = ftell(self.fp);  # get position of EOF
        num_records = (eof - timing_data_starts) // 4;  # 4 bytes (32 bits) per timing record

        fseek(self.fp, 3584 + 48, SEEK_SET); # go to the start of the timing data
        cdef unsigned int record

        cdef unsigned int last
        last_channel = None
        first = False

        time_deltas = []
        for i in range(num_records):
            fread(&record, sizeof(record), 1, self.fp)
            time = record & 0x0FFFFFFF  # Extracting the first 28 bits
            channel = (record & 0xF0000000) >> 28  # Extracting the last 4 bits

            if channel == 0xF:
                markers = time & 0xF
                if markers == 0:
                    self.overflow_correction += TIME_OVERFLOW
            else:
                if channel not in self.photon_timestamps:
                    self.photon_timestamps[channel] = []
                true_time = (self.overflow_correction + time) * 4e-12

                if channel != last_channel:
                    if not first:
                        time_deltas.append(true_time - last)
                    last_channel = channel
                    last = true_time
                    first = False
        return time_deltas


    def read_all(self):
        timing_data_starts = 3584 + 48;
        fseek(self.fp, 0, SEEK_END);  # move to end of file
        cdef long int eof = ftell(self.fp);  # get position of EOF
        num_records = (eof - timing_data_starts) // 4;  # 4 bytes (32 bits) per timing record

        fseek(self.fp, 3584 + 48, SEEK_SET); # go to the start of the timing data
        cdef unsigned int record

        for i in range(num_records):
            fread(&record, sizeof(record), 1, self.fp)
            time = record & 0x0FFFFFFF  # Extracting the first 28 bits
            channel = (record & 0xF0000000) >> 28  # Extracting the last 4 bits

            if channel == 0xF:
                markers = time & 0xF
                if markers == 0:
                    self.overflow_correction += TIME_OVERFLOW
            else:
                if channel not in self.photon_timestamps:
                    self.photon_timestamps[channel] = []
                true_time = self.overflow_correction + time
                self.photon_timestamps[channel].append(true_time * 4e-12)

        return self.photon_timestamps

    def n(self):
        return len(self.photon_timestamps)

    def __dealloc__(self):
        if self.fp is not NULL:
            fclose(self.fp)

    def event(self):
        cdef unsigned int record
        cdef unsigned int time, channel
        cdef long long true_time

        if fread(&record, sizeof(record), 1, self.fp) != 1:
            return None  # End of File or error

        # Manually extract time and channel from the 32-bit integer
        time = record & 0x0FFFFFFF  # Extracting the first 28 bits
        channel = (record & 0xF0000000) >> 28  # Extracting the last 4 bits

        if channel == 0xF:
            markers = time & 0xF
            if markers == 0:
                self.overflow_correction += 210698240
            return None
        else:
            true_time = self.overflow_correction + time
            return {"channel": channel, "time": true_time}
