# cython: language_level=3, binding=True, boundscheck=False, wraparound=False, nonecheck=False
# distutils: language=c++

from libc.stdio cimport FILE, fopen, fread, fseek, fclose, SEEK_SET
from libc.string cimport memset
from libc.stdint cimport int32_t, int64_t, uint32_t
from libc.stdlib cimport malloc, free, realloc
from libcpp.map cimport map
from libcpp.vector cimport vector
from libc.math cimport fmin, fmax

import time
import struct
import numpy as np

cdef long long TIME_OVERFLOW = 210698240

# Type tag constants for PTU header parsing (computed once at module load)
_TY_EMPTY8      = struct.unpack(">i", bytes.fromhex("FFFF0008"))[0]
_TY_BOOL8       = struct.unpack(">i", bytes.fromhex("00000008"))[0]
_TY_INT8        = struct.unpack(">i", bytes.fromhex("10000008"))[0]
_TY_BITSET64    = struct.unpack(">i", bytes.fromhex("11000008"))[0]
_TY_COLOR8      = struct.unpack(">i", bytes.fromhex("12000008"))[0]
_TY_FLOAT8      = struct.unpack(">i", bytes.fromhex("20000008"))[0]
_TY_TDATETIME   = struct.unpack(">i", bytes.fromhex("21000008"))[0]
_TY_FLOAT8ARRAY = struct.unpack(">i", bytes.fromhex("2001FFFF"))[0]
_TY_ANSISTRING  = struct.unpack(">i", bytes.fromhex("4001FFFF"))[0]
_TY_WIDESTRING  = struct.unpack(">i", bytes.fromhex("4002FFFF"))[0]
_TY_BINARYBLOB  = struct.unpack(">i", bytes.fromhex("FFFFFFFF"))[0]

# Record type constants
_RT_PICOHARP_T2 = struct.unpack(">i", bytes.fromhex("00010203"))[0]
_RT_PICOHARP_T3 = struct.unpack(">i", bytes.fromhex("00010303"))[0]



def read_header(path):
    with open(path, "rb") as inputfile:
        magic = inputfile.read(8).decode("utf-8").strip("\0")
        if magic != "PQTTTR":
            raise ValueError("ERROR: Magic invalid, this is not a PTU file.")

        version = inputfile.read(8).decode("utf-8").strip("\0")

        tagDataList = []  # Contains tuples of (tagName, tagValue)
        for i in range(400):
            tagIdent = inputfile.read(32).decode("utf-8").strip("\0")
            tagIdx = struct.unpack("<i", inputfile.read(4))[0]
            tagTyp = struct.unpack("<i", inputfile.read(4))[0]
            if tagIdx > -1:
                evalName = tagIdent + "(" + str(tagIdx) + ")"
            else:
                evalName = tagIdent

            if tagTyp == _TY_EMPTY8:
                inputfile.read(8)
                tagDataList.append((evalName, "<empty Tag>"))
            elif tagTyp == _TY_BOOL8:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                tagDataList.append((evalName, "False" if tagInt == 0 else "True"))
            elif tagTyp == _TY_INT8:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                tagDataList.append((evalName, tagInt))
            elif tagTyp == _TY_BITSET64:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                tagDataList.append((evalName, tagInt))
            elif tagTyp == _TY_COLOR8:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                tagDataList.append((evalName, tagInt))
            elif tagTyp == _TY_FLOAT8:
                tagFloat = struct.unpack("<d", inputfile.read(8))[0]
                tagDataList.append((evalName, tagFloat))
            elif tagTyp == _TY_FLOAT8ARRAY:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                tagDataList.append((evalName, tagInt))
            elif tagTyp == _TY_TDATETIME:
                tagFloat = struct.unpack("<d", inputfile.read(8))[0]
                tagTime = int((tagFloat - 25569) * 86400)
                tagTime = time.gmtime(tagTime)
                tagDataList.append((evalName, tagTime))
            elif tagTyp == _TY_ANSISTRING:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                tagString = inputfile.read(tagInt).decode("utf-8").strip("\0")
                tagDataList.append((evalName, tagString))
            elif tagTyp == _TY_WIDESTRING:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                tagString = (
                    inputfile.read(tagInt).decode("utf-16le", errors="ignore").strip("\0")
                )
                tagDataList.append((evalName, tagString))
            elif tagTyp == _TY_BINARYBLOB:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                tagDataList.append((evalName, tagInt))
            else:
                raise Exception("ERROR: Unknown tag type")
            if tagIdent == "Header_End":
                break

        tagNames = [tagDataList[i][0] for i in range(len(tagDataList))]
        tagValues = [tagDataList[i][1] for i in range(len(tagDataList))]
        tags_end = inputfile.tell()

    return dict(zip(tagNames, tagValues)), tags_end


def t3_to_histogram(
    str path,
    double min_ns = -1,
    double max_ns = 0xFFFFFFFF,
    window_ends_sec = [],
    int segments = 1,
):
    if segments < 1:
        raise ValueError("Cannot have < 1 segments.")
    if window_ends_sec and segments > 1:
        raise ValueError("Cannot specify both `window_ends_sec` and `segments` simultaneously.")
    if min_ns > max_ns:
        raise ValueError("min_dtime cannot be larger than max_dtime.")

    tags, header_end = read_header(path)

    rec_type = tags["TTResultFormat_TTTRRecType"]
    if rec_type != _RT_PICOHARP_T3:
        raise ValueError(f"Unsupported record type {rec_type:#010x}. Only PicoHarp T3 is supported.")

    cdef FILE *fp = fopen(path.encode('utf-8'), "rb")
    fseek(fp, header_end, SEEK_SET)

    bin_size_ps = tags["MeasDesc_Resolution"] * 1e12
    sync_rate_Hz = tags["TTResult_SyncRate"]
    measurement_runtime_sec = tags["TTResult_StopAfter"] / 1000

    if window_ends_sec:
        last_we = window_ends_sec[len(window_ends_sec) - 1]
        if last_we < measurement_runtime_sec:
            remaining_time = measurement_runtime_sec - last_we
            print(f"Warning: some measurement time left unchecked: ({remaining_time:.1f} sec)")
    else:
        window_ends_sec = [(i + 1) * measurement_runtime_sec // segments for i in range(segments)]

    window_ends_nsync = [we * sync_rate_Hz for we in window_ends_sec]
    n_we = len(window_ends_sec)
    window_starts_sec = [0] + window_ends_sec[:n_we - 1]

    histogrammers = [
        T3Histogrammer(
            bin_size_ps=bin_size_ps,
            sync_rate_Hz=sync_rate_Hz,
            min_ns=min_ns,
            max_ns=max_ns,
            start_sec=ws,
            stop_sec=we,
        )
        for ws, we in zip(window_starts_sec, window_ends_sec)
    ]

    cdef uint32_t record
    cdef size_t record_size = sizeof(record)
    cdef size_t read_count
    cdef char *buff = <char *>malloc(record_size * 1_000_000)
    cdef uint32_t channel
    cdef uint32_t dtime
    cdef uint32_t nsync
    cdef long long overflow_correction = 0
    cdef long long true_nsync = 0
    cdef int current_hist_idx = 0
    cdef int n_windows = len(window_ends_nsync)
    cdef bint done = False

    while not done:
        read_count = fread(buff, record_size, 1_000_000, fp)

        if read_count == 0:  # hit EOF
            break

        for i in range(read_count):
            record = (<uint32_t *>buff)[i]

            channel = (record & 0xF0000000)
            dtime = (record & 0x0FFF0000) >> 16
            nsync = (record & 0x0000FFFF)

            if channel == 0xF0000000:  # special record
                if dtime == 0:  # overflow
                    overflow_correction += 65536
                # marker records: no photon to record
            else:
                true_nsync = overflow_correction + nsync

                # This assumes the windows are wide enough that each will get at least 1 photon.
                # That's a very reasonable assumption, but it's possible to violate it!
                if true_nsync > window_ends_nsync[current_hist_idx]:
                    if current_hist_idx == n_windows - 1:
                        done = True
                        break
                    current_hist_idx += 1

                histogrammers[current_hist_idx]._click(dtime)

    free(buff)
    fclose(fp)

    if len(histogrammers) == 1:
        return histogrammers[0].times_ns, histogrammers[0].counts

    return histogrammers


cdef class T3Histogrammer:
    cdef public double bin_size_ns
    cdef public double min_bin, max_bin
    cdef uint32_t *counts_arr
    cdef public int array_size

    cdef double overflow_correction
    cdef public double true_nsync
    cdef public int sync_rate_Hz

    cdef public start_sec
    cdef double start_nsync
    cdef public stop_sec
    cdef double stop_nsync

    def __cinit__(self, bin_size_ps, sync_rate_Hz, double min_ns=-1, double max_ns=0xFFFFFFFF,
            double start_sec=0, double stop_sec=0xFFFFFFFF,
        ):
        self.bin_size_ns = bin_size_ps / 1000  # change to nanoseconds
        # convert max/min from nanoseconds to bin number
        self.min_bin, self.max_bin = min_ns * 1000 / bin_size_ps, max_ns * 1000 / bin_size_ps
        # Determine the size of the counts array based on sync_rate.
        # The latest timestamp we should get is equal to the sync period.
        self.array_size = int(1e9 / (sync_rate_Hz * self.bin_size_ns))
        self.counts_arr = <uint32_t*>malloc(self.array_size * sizeof(uint32_t))
        if self.counts_arr is NULL:
            raise MemoryError("Failed to allocate memory for counts array.")
        memset(self.counts_arr, 0, self.array_size * sizeof(uint32_t))

        self.true_nsync = 0
        self.overflow_correction = 0
        self.sync_rate_Hz = sync_rate_Hz

        # Allows windowing: only record events falling between `start_sec` and `stop_sec`
        self.start_sec = start_sec
        self.start_nsync = start_sec * sync_rate_Hz
        self.stop_sec = stop_sec
        self.stop_nsync = stop_sec * sync_rate_Hz

    def __dealloc__(self):
        if self.counts_arr is not NULL:
            free(self.counts_arr)

    property times_ns:
        def __get__(self):
            cdef int start = max(0, <int>self.min_bin) if self.min_bin > 0 else 0
            cdef int stop = self.array_size if self.max_bin >= self.array_size else <int>self.max_bin + 1
            return np.arange(start, stop, dtype=np.float64) * self.bin_size_ns

    property counts:
        def __get__(self):
            cdef int start = max(0, <int>self.min_bin) if self.min_bin > 0 else 0
            cdef int stop = self.array_size if self.max_bin >= self.array_size else <int>self.max_bin + 1
            cdef uint32_t[:] mv = <uint32_t[:self.array_size]>self.counts_arr
            return np.asarray(mv)[start:stop].copy()

    property runtime_sec:
        def __get__(self):
            return self.true_nsync / self.sync_rate_Hz

    property count_rate:
        def __get__(self):
            cdef uint32_t total = 0
            cdef int i
            for i in range(self.array_size):
                total += self.counts_arr[i]
            return total / self.runtime_sec

    def click(self, uint32_t record):
        cdef uint32_t channel = (record & 0xF0000000)
        cdef uint32_t dtime = (record & 0x0FFF0000) >> 16
        cdef uint32_t nsync = (record & 0x0000FFFF)

        if channel == 0xF0000000:  # special record
            if dtime == 0:  # overflow
                self.overflow_correction += 65536
            else:
                self.true_nsync = self.overflow_correction + nsync
        else:
            self.true_nsync = self.overflow_correction + nsync
            if (self.true_nsync < self.start_nsync) or (self.stop_nsync < self.true_nsync):
                return
            self._click(dtime)

    def _click(self, uint32_t dtime):
        cdef uint32_t *new_counts_arr
        cdef int new_size

        if (dtime < self.min_bin) or (self.max_bin < dtime):
            return

        if dtime < <uint32_t>self.array_size:
            self.counts_arr[dtime] += 1
        else:
            # Grow the array enough to fit dtime, with some headroom
            new_size = max(int(self.array_size * 1.01), <int>dtime + 1)

            new_counts_arr = <uint32_t *>realloc(self.counts_arr, new_size * sizeof(uint32_t))
            if new_counts_arr is NULL:
                raise MemoryError("Memory allocation error.")

            memset(new_counts_arr + self.array_size, 0, (new_size - self.array_size) * sizeof(uint32_t))
            self.counts_arr = new_counts_arr
            self.array_size = new_size
            self.counts_arr[dtime] += 1

    def batch(self, records):
        for record in records:
            self.click(record)

    def clear(self):
        memset(self.counts_arr, 0, self.array_size * sizeof(uint32_t))
        self.true_nsync = 0
        self.overflow_correction = 0


def t2_to_timestamps(str path):
    tags, header_end = read_header(path)

    rec_type = tags["TTResultFormat_TTTRRecType"]
    if rec_type != _RT_PICOHARP_T2:
        raise ValueError(f"Unsupported record type {rec_type:#010x}. Only PicoHarp T2 is supported.")

    resolution_ps = tags["MeasDesc_GlobalResolution"] * 1e12

    cdef FILE *fp = fopen(path.encode('utf-8'), "rb")
    fseek(fp, header_end, SEEK_SET)

    processor = T2Streamer(resolution_ps)

    cdef uint32_t record
    cdef size_t record_size = sizeof(record)
    cdef size_t read_count
    cdef char *buff = <char *>malloc(record_size * 1_000_000)

    while True:
        read_count = fread(buff, record_size, 1_000_000, fp)
        if read_count == 0:  # hit EOF
            break

        for i in range(read_count):
            record = (<uint32_t *>buff)[i]
            processor.click(record)

    free(buff)
    fclose(fp)
    return processor.ch0_times_ns, processor.ch1_times_ns


cdef class T2Streamer:
    cdef size_t ch0_max_timestamps
    cdef size_t ch1_max_timestamps
    cdef long long overflow_correction
    cdef long long *ch0_photon_timestamps
    cdef long long *ch1_photon_timestamps
    cdef size_t ch0_timestamp_index
    cdef size_t ch1_timestamp_index
    cdef public double resolution_ns

    def __cinit__(self, resolution_ps):
        self.ch0_max_timestamps = 1_000_000
        self.ch1_max_timestamps = 1_000_000
        self.ch0_photon_timestamps = <long long *>malloc(self.ch0_max_timestamps * sizeof(long long))
        self.ch1_photon_timestamps = <long long *>malloc(self.ch1_max_timestamps * sizeof(long long))
        self.ch0_timestamp_index = 0
        self.ch1_timestamp_index = 0
        self.overflow_correction = 0
        self.resolution_ns = resolution_ps / 1_000

    def __dealloc__(self):
        if self.ch0_photon_timestamps is not NULL:
            free(self.ch0_photon_timestamps)
        if self.ch1_photon_timestamps is not NULL:
            free(self.ch1_photon_timestamps)

    def clear(self):
        memset(self.ch0_photon_timestamps, 0, self.ch0_max_timestamps * sizeof(long long))
        memset(self.ch1_photon_timestamps, 0, self.ch1_max_timestamps * sizeof(long long))
        self.ch0_timestamp_index = 0
        self.ch1_timestamp_index = 0
        self.ch0_max_timestamps = 1_000_000
        self.ch1_max_timestamps = 1_000_000
        self.overflow_correction = 0

    def click(self, uint32_t record):
        cdef unsigned int time, channel
        cdef long long true_time
        cdef size_t new_size

        time = record & 0x0FFFFFFF      # first 28 bits
        channel = (record & 0xF0000000) >> 28  # last 4 bits

        if channel == 0xF:
            if (time & 0xF) == 0:
                self.overflow_correction += TIME_OVERFLOW  # wraparound
            return

        true_time = self.overflow_correction + time
        if channel == 0:
            if self.ch0_timestamp_index == self.ch0_max_timestamps - 1:
                new_size = int(1.1 * self.ch0_max_timestamps)
                self.ch0_photon_timestamps = <long long *>realloc(self.ch0_photon_timestamps, new_size * sizeof(long long))
                self.ch0_max_timestamps = new_size
            self.ch0_photon_timestamps[self.ch0_timestamp_index] = true_time
            self.ch0_timestamp_index += 1
        elif channel == 1:
            if self.ch1_timestamp_index == self.ch1_max_timestamps - 1:
                new_size = int(1.1 * self.ch1_max_timestamps)
                self.ch1_photon_timestamps = <long long *>realloc(self.ch1_photon_timestamps, new_size * sizeof(long long))
                self.ch1_max_timestamps = new_size
            self.ch1_photon_timestamps[self.ch1_timestamp_index] = true_time
            self.ch1_timestamp_index += 1

    def batch(self, records):
        for record in records:
            self.click(record)

    property ch0_times_ns:
        def __get__(self):
            if self.ch0_timestamp_index == 0:
                return np.array([], dtype=np.float64)
            cdef long long[:] mv = <long long[:self.ch0_timestamp_index]>self.ch0_photon_timestamps
            return np.asarray(mv) * self.resolution_ns

    property ch1_times_ns:
        def __get__(self):
            if self.ch1_timestamp_index == 0:
                return np.array([], dtype=np.float64)
            cdef long long[:] mv = <long long[:self.ch1_timestamp_index]>self.ch1_photon_timestamps
            return np.asarray(mv) * self.resolution_ns

    property ch0_count:
        def __get__(self):
            return self.ch0_timestamp_index

    property ch1_count:
        def __get__(self):
            return self.ch1_timestamp_index


cdef class T2TimestampIterator:
    """
    Simple iterator for T2 files that reads and yields (channel, timestamp_ns) tuples.
    """
    cdef FILE *fp
    cdef char *buff
    cdef size_t read_position
    cdef size_t read_count
    cdef long long overflow_correction
    cdef double resolution_ns

    def __cinit__(self, str path):
        tags, header_end = read_header(path)

        rec_type = tags.get("TTResultFormat_TTTRRecType")
        if rec_type != _RT_PICOHARP_T2:
            raise ValueError(f"Unsupported record type {rec_type:#010x}. Only PicoHarp T2 is supported.")

        self.resolution_ns = tags.get("MeasDesc_GlobalResolution", 0) * 1e9

        self.fp = fopen(path.encode('utf-8'), "rb")
        if self.fp == NULL:
            raise IOError(f"Could not open file: {path}")

        fseek(self.fp, header_end, SEEK_SET)

        self.buff = <char *>malloc(sizeof(uint32_t) * 1_000_000)
        if self.buff == NULL:
            fclose(self.fp)
            raise MemoryError("Could not allocate buffer")

        self.read_position = 0
        self.read_count = 0
        self.overflow_correction = 0

    def __dealloc__(self):
        if self.fp != NULL:
            fclose(self.fp)
        if self.buff != NULL:
            free(self.buff)

    def __iter__(self):
        return self

    def __next__(self):
        cdef uint32_t record
        cdef unsigned int time, channel
        cdef long long true_time

        while True:
            if self.read_position >= self.read_count:
                self.read_count = fread(self.buff, sizeof(uint32_t), 1_000_000, self.fp)
                self.read_position = 0
                if self.read_count == 0:
                    raise StopIteration

            record = (<uint32_t *>self.buff)[self.read_position]
            self.read_position += 1

            time = record & 0x0FFFFFFF
            channel = (record & 0xF0000000) >> 28

            if channel == 0xF:  # special record
                if (time & 0xF) == 0:  # overflow
                    self.overflow_correction += TIME_OVERFLOW
                continue

            true_time = self.overflow_correction + time
            return channel, true_time * self.resolution_ns


def t2_iterator(path):
    """Create and return a T2TimestampIterator instance"""
    return T2TimestampIterator(path)


cdef class G2:
    cdef int n_bins
    cdef vector[double] bin_edges
    cdef vector[vector[double]] bin
    cdef vector[double] num_pairs
    cdef double min_timestamp
    cdef double max_timestamp
    cdef dict data

    def __init__(self, list bin_edges):
        cdef int i
        self.n_bins = len(bin_edges) - 1
        self.bin_edges = vector[double]()
        for edge in bin_edges:
            self.bin_edges.push_back(edge)
        self.bin = vector[vector[double]](self.n_bins)
        self.num_pairs = vector[double](self.n_bins, 0.0)
        self.min_timestamp = float('inf')
        self.max_timestamp = float('-inf')
        self.data = {'a': [], 'b': []}

    @property
    def normalization(self):
        cdef double duration = self.max_timestamp - self.min_timestamp
        cdef int i, j, count_a, count_b
        cdef double tau, max_b, denom
        cdef int n_a = len(self.data['a'])
        cdef int n_b = len(self.data['b'])
        cdef vector[double] a_vec = vector[double](n_a)
        cdef vector[double] b_vec = vector[double](n_b)

        for i in range(n_a):
            a_vec[i] = self.data['a'][i]
        for i in range(n_b):
            b_vec[i] = self.data['b'][i]

        max_b = b_vec.back() if not b_vec.empty() else 0.0

        result = []
        for i in range(self.n_bins):
            tau = self.bin_edges[i + 1]
            count_a = 0
            count_b = 0
            for j in range(<int>a_vec.size()):
                if a_vec[j] >= tau:
                    count_a += 1
            for j in range(<int>b_vec.size()):
                if b_vec[j] <= (max_b - tau):
                    count_b += 1
            denom = count_a * count_b
            result.append((duration - tau) / denom if denom != 0 else 0.0)
        return result

    @property
    def g2(self):
        G2_vals = self.G2
        norms = self.normalization
        return [G2_vals[i] * norms[i] for i in range(self.n_bins)]

    @property
    def G2(self):
        return [
            self.num_pairs[i] / (self.bin_edges[i+1] - self.bin_edges[i])
            for i in range(self.n_bins)
        ]

    cpdef void click(self, str channel, double timestamp):
        cdef int i, j
        cdef double stop, oldest_allowed
        cdef vector[double] too_old

        self.min_timestamp = fmin(timestamp, self.min_timestamp)
        self.max_timestamp = fmax(timestamp, self.max_timestamp)
        self.data[channel].append(timestamp)

        if channel == 'b':
            for i in range(self.n_bins):
                stop = self.bin_edges[i+1]
                oldest_allowed = timestamp - stop
                too_old.clear()

                for j in range(<int>self.bin[i].size()):
                    if self.bin[i][j] <= oldest_allowed:
                        too_old.push_back(self.bin[i][j])

                if not too_old.empty():
                    self.bin[i].erase(
                        self.bin[i].begin(),
                        self.bin[i].begin() + too_old.size()
                    )

                    if i < self.n_bins - 1:
                        self.bin[i + 1].insert(
                            self.bin[i + 1].end(),
                            too_old.begin(),
                            too_old.end()
                        )

            for i in range(self.n_bins):
                self.num_pairs[i] += self.bin[i].size()

        if channel == 'a':
            self.bin[0].push_back(timestamp)


cdef double pnormalize(double* G, double* t, double* u, double* bins, int n_t, int n_u, int n_bins):
    cdef double duration = fmax(t[n_t-1], u[n_u-1]) - fmin(t[0], u[0])
    cdef double* Gn = <double*>malloc(n_bins * sizeof(double))
    cdef int i, j
    cdef double tau
    cdef int t_count, u_count
    cdef double u_max = u[n_u-1]

    for i in range(n_bins):
        Gn[i] = G[i]

    for i in range(1, n_bins + 1):
        tau = bins[i]
        t_count = 0
        u_count = 0
        for j in range(n_t):
            if t[j] >= tau:
                t_count += 1
        for j in range(n_u):
            if u[j] <= (u_max - tau):
                u_count += 1
        Gn[i-1] *= ((duration - tau) / (t_count * u_count))

    for i in range(n_bins):
        G[i] = Gn[i]

    free(Gn)
    return duration


cdef void pcorrelate_impl(double* t, double* u, double* bins, long long* counts,
                          int n_t, int n_u, int n_bins):
    cdef int i, j, k
    cdef double ti, tau_min, tau_max
    cdef int* imin = <int*>malloc(n_bins * sizeof(int))
    cdef int* imax = <int*>malloc(n_bins * sizeof(int))

    for i in range(n_bins):
        imin[i] = 0
        imax[i] = 0
        counts[i] = 0

    for i in range(n_t):
        ti = t[i]
        for k in range(n_bins):
            tau_min = bins[k]
            tau_max = bins[k+1]

            if k == 0:
                j = imin[k]
                while j < n_u:
                    if u[j] - ti >= tau_min:
                        break
                    j += 1

            imin[k] = j
            if imax[k] > j:
                j = imax[k]
            while j < n_u:
                if u[j] - ti >= tau_max:
                    break
                j += 1
            imax[k] = j

        for k in range(n_bins):
            counts[k] += imax[k] - imin[k]

    free(imin)
    free(imax)


cdef double* list_to_array(list py_list, int* size):
    cdef int i, n = len(py_list)
    cdef double* arr = <double*>malloc(n * sizeof(double))
    for i in range(n):
        arr[i] = py_list[i]
    size[0] = n
    return arr


def pcorrelate(list t, list u, list bins, bint normalize=False):
    cdef int n_t, n_u, n_bins
    cdef double *t_arr = list_to_array(t, &n_t)
    cdef double *u_arr = list_to_array(u, &n_u)
    cdef double *bins_arr = list_to_array(bins, &n_bins)
    n_bins -= 1  # number of bins is one less than number of bin edges

    cdef long long* counts = <long long*>malloc(n_bins * sizeof(long long))
    cdef double* G = <double*>malloc(n_bins * sizeof(double))
    cdef int i

    pcorrelate_impl(t_arr, u_arr, bins_arr, counts, n_t, n_u, n_bins)

    for i in range(n_bins):
        G[i] = counts[i] / (bins_arr[i+1] - bins_arr[i])

    if normalize:
        pnormalize(G, t_arr, u_arr, bins_arr, n_t, n_u, n_bins)

    cdef list result = [G[i] for i in range(n_bins)]

    free(t_arr)
    free(u_arr)
    free(bins_arr)
    free(counts)
    free(G)

    return result
