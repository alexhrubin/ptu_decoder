# cython: language_level=3
# distutils: language=c++

from libc.stdio cimport FILE, fopen, fread, fseek, fclose, SEEK_SET, SEEK_END, ftell, perror, SEEK_CUR
from libc.string cimport memcmp, memset
from libc.stdint cimport int32_t, int64_t, uint32_t

from libc.stdlib cimport malloc, free, realloc
from libc.stdio cimport fopen, fread, fclose, FILE
from libcpp.map cimport map

import time
import struct

cdef long long TIME_OVERFLOW = 210698240


cdef void check_magic_string(FILE *fp) except *:  # except * allows propagating C exceptions as Python exceptions
    cdef char header[8]  # Create a buffer to store the header (magic number)
    
    if fread(header, 1, 8, fp) != 8:  # Read 8 bytes from the file into the header buffer
        fclose(fp)
        perror("Failed to read the header from the file")
        raise IOError("Failed to read the header from the file")

    # Check if the header matches "PQTTTR"
    if memcmp(header, b"PQTTTR\0\0", 8) != 0:
        fclose(fp)
        raise ValueError("ERROR: Magic invalid, this is not a PTU file.")


def read_header(path):
    tyEmpty8 = struct.unpack(">i", bytes.fromhex("FFFF0008"))[0]
    tyBool8 = struct.unpack(">i", bytes.fromhex("00000008"))[0]
    tyInt8 = struct.unpack(">i", bytes.fromhex("10000008"))[0]
    tyBitSet64 = struct.unpack(">i", bytes.fromhex("11000008"))[0]
    tyColor8 = struct.unpack(">i", bytes.fromhex("12000008"))[0]
    tyFloat8 = struct.unpack(">i", bytes.fromhex("20000008"))[0]
    tyTDateTime = struct.unpack(">i", bytes.fromhex("21000008"))[0]
    tyFloat8Array = struct.unpack(">i", bytes.fromhex("2001FFFF"))[0]
    tyAnsiString = struct.unpack(">i", bytes.fromhex("4001FFFF"))[0]
    tyWideString = struct.unpack(">i", bytes.fromhex("4002FFFF"))[0]
    tyBinaryBlob = struct.unpack(">i", bytes.fromhex("FFFFFFFF"))[0]

    inputfile = open(path, "rb")

    magic = inputfile.read(8).decode("utf-8").strip("\0")
    if magic != "PQTTTR":
        raise Exception("ERROR: Magic invalid, this is not a PTU file.")
        inputfile.close()

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

        if tagTyp == tyEmpty8:
            inputfile.read(8)
            tagDataList.append((evalName, "<empty Tag>"))
        elif tagTyp == tyBool8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            if tagInt == 0:
                tagDataList.append((evalName, "False"))
            else:
                tagDataList.append((evalName, "True"))
        elif tagTyp == tyInt8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyBitSet64:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyColor8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyFloat8:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
            tagDataList.append((evalName, tagFloat))
        elif tagTyp == tyFloat8Array:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyTDateTime:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
            tagTime = int((tagFloat - 25569) * 86400)
            tagTime = time.gmtime(tagTime)
            tagDataList.append((evalName, tagTime))
        elif tagTyp == tyAnsiString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode("utf-8").strip("\0")
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyWideString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = (
                inputfile.read(tagInt).decode("utf-16le", errors="ignore").strip("\0")
            )
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyBinaryBlob:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        else:
            raise Exception("ERROR: Unknown tag type")
        if tagIdent == "Header_End":
            break


    tagNames = [tagDataList[i][0] for i in range(0, len(tagDataList))]
    tagValues = [tagDataList[i][1] for i in range(0, len(tagDataList))]
    tags_end = inputfile.tell()

    inputfile.close()

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
    num_records = tags["TTResult_NumberOfRecords"]
    resolution = tags["MeasDesc_Resolution"] * 1e9  # in nanoseconds

    cdef FILE *fp = fopen(path.encode('utf-8'), "rb")
    fseek(fp, header_end, SEEK_SET)

    bin_size_ps = tags["MeasDesc_Resolution"] * 1e12
    sync_rate_Hz = tags["TTResult_SyncRate"]
    measurement_runtime_sec = tags["TTResult_StopAfter"] / 1000

    if window_ends_sec:
        # alert the user if the user-supplied window ends don't capture all available data
        if window_ends_sec[-1] < measurement_runtime_sec:
            remaining_time = measurement_runtime_sec - window_ends_sec[-1]
            print(f"Warning: some measurement time left unchecked: ({remaining_time:.1f} sec)")
    else:
        window_ends_sec = [(i + 1) * measurement_runtime_sec // segments for i in range(segments)]

    window_ends_nsync = [we * sync_rate_Hz for we in window_ends_sec]

    window_starts_sec = [0] + window_ends_sec[:-1]
    
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
    cdef size_t buffer_size = record_size * 1_000_000
    cdef size_t read_count
    buff = <char *>malloc(buffer_size)

    cdef overflow_correction = 0
    cdef true_nsync = 0
    cdef uint32_t channel
    cdef uint32_t dtime
    cdef uint32_t nsync
    cdef current_hist_idx = 0

    while True:
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
                else:
                    true_nsync = overflow_correction + nsync
            else:
                true_nsync = overflow_correction + nsync
            
            # This assumes the windows are wide enough that each will get at least 1 photon
            # That's a very reasonable assumption, but it's possible to violate it!
            if true_nsync > window_ends_nsync[current_hist_idx]:
                if current_hist_idx == len(window_ends_nsync) - 1:
                    break
                current_hist_idx += 1
            
            histogrammers[current_hist_idx]._click(dtime)
            
    fclose(fp)

    if len(histogrammers) == 1:
        return histogrammers[0].times_ns, histogrammers[0].counts

    return histogrammers


cdef class T3Histogrammer:
    cdef public double bin_size_ns
    cdef public double min_bin, max_bin
    cdef uint32_t *counts_arr
    cdef uint32_t *new_counts_arr
    cdef public int array_size
    cdef int new_size

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
        # Determine the size of the counts array based on sync_rate
        # The latest timestamp we should get is equal to the sync period
        self.array_size = int(1e9 / (sync_rate_Hz * self.bin_size_ns))
        self.counts_arr = <uint32_t*>malloc(self.array_size * sizeof(uint32_t))
        if self.counts_arr is NULL:
            raise MemoryError("Failed to allocate memory for counts array.")
        # Initialize array with zeros
        memset(self.counts_arr, 0, self.array_size * sizeof(uint32_t))

        self.true_nsync = 0
        self.overflow_correction = 0
        self.sync_rate_Hz = sync_rate_Hz

        # Allows windowing: only record events falling between `start_sec` and `stop_sec`
        self.start_sec = start_sec
        self.start_nsync = start_sec * sync_rate_Hz
        self.stop_sec = stop_sec
        self.stop_nsync = stop_sec * sync_rate_Hz

    property times_ns:
        def __get__(self):
            return [i * self.bin_size_ns for i in range(self.array_size) if (self.min_bin <= i) and (i <= self.max_bin)]

    property counts:
        def __get__(self):
            return [self.counts_arr[i] for i in range(self.array_size) if (self.min_bin <= i) and (i <= self.max_bin)]

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
                # We're outside the time range that we care about, so ignore this event
                return

            self._click(dtime)

    def _click(self, uint32_t dtime):
        if (dtime < self.min_bin) or (self.max_bin < dtime):
            return  # return early if out of desired range

        if dtime < self.array_size:
            self.counts_arr[dtime] += 1
        else:
            # Calculate new size (1% larger)
            new_size = int(self.array_size * 1.01)

            # Allocate new memory block
            new_counts_arr = <uint32_t *>realloc(self.counts_arr, new_size * sizeof(uint32_t))
            if new_counts_arr is NULL:
                # Handle allocation failure
                raise Exception("Memory allocation error.")
            else:
                # Zero out the new portion
                memset(new_counts_arr + self.array_size, 0, (new_size - self.array_size) * sizeof(uint32_t))

                # Update the pointer and size
                self.counts_arr = new_counts_arr
                self.array_size = new_size

            self.counts_arr[dtime] += 1

    def batch(self, records):
        for record in records:
            self.click(record)

    def clear(self):
        # reinitialize array with zeros
        memset(self.counts_arr, 0, self.array_size * sizeof(uint32_t))
        self.true_nsync = 0
        self.overflow_correction = 0


def t2_to_timestamps(str path):
    tags, header_end = read_header(path)
    num_records = tags["TTResult_NumberOfRecords"]
    resolution = tags["MeasDesc_Resolution"] * 1e9  # in nanoseconds

    cdef FILE *fp = fopen(path.encode('utf-8'), "rb")
    fseek(fp, header_end, SEEK_SET)

    resolution_ps = tags["MeasDesc_Resolution"] * 1e12

    processor = T2Streamer(resolution_ps)

    cdef uint32_t dtime
    cdef uint32_t record
    cdef size_t record_size = sizeof(record)
    cdef size_t buffer_size = record_size * 1_000_000
    cdef size_t read_count

    buff = <char *>malloc(buffer_size)

    while True:
        read_count = fread(buff, record_size, 1_000_000, fp)
        if read_count == 0:  # hit EOF
            break

        for i in range(read_count):
            record = (<uint32_t *>buff)[i]
            processor.click(record)

    fclose(fp)
    return processor.ch0_times_ns, processor.ch1_times_ns


cdef class T2Streamer:
    cdef size_t ch0_max_timestamps
    cdef size_t ch1_max_timestamps
    cdef long long overflow_correction
    cdef unsigned int num_events
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

    def clear(self):
        # reinitialize array with zeros
        memset(self.ch0_photon_timestamps, 0, self.ch0_max_timestamps * sizeof(uint32_t))
        memset(self.ch1_photon_timestamps, 0, self.ch1_max_timestamps * sizeof(uint32_t))
        self.ch0_timestamp_index = 0
        self.ch1_timestamp_index = 0
        self.ch0_max_timestamps = 1_000_000
        self.ch1_max_timestamps = 1_000_000
        self.overflow_correction = 0

    def click(self, uint32_t record):
        cdef unsigned int time, channel
        cdef long long true_time

        # Manually extract time and channel from the 32-bit integer
        time = record & 0x0FFFFFFF  # Extracting the first 28 bits
        channel = (record & 0xF0000000) >> 28  # Extracting the last 4 bits

        if channel == 0xF:
            markers = time & 0xF
            if markers == 0:
                self.overflow_correction += TIME_OVERFLOW  # wraparound
            return None
        else:
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
            else:
                # This is for special record types, which currently we just ignore
                pass

    def batch(self, records):
        for record in records:
            self.click(record)

    property ch0_times_ns:
        def __get__(self):
            return [self.ch0_photon_timestamps[i] * self.resolution_ns for i in range(self.ch0_timestamp_index)]

    property ch1_times_ns:
        def __get__(self):
            return [self.ch1_photon_timestamps[i] * self.resolution_ns for i in range(self.ch1_timestamp_index)]

    property ch0_count:
        def __get__(self):
            return self.ch0_timestamp_index

    property ch1_count:
        def __get__(self):
            return self.ch1_timestamp_index


###########
# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from libc.math cimport fmin, fmax

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
        self.min_timestamp = 0
        self.max_timestamp = -1
        self.data = {'a': [], 'b': []}
    
    @property
    def normalization(self):
        cdef double duration = self.max_timestamp - self.min_timestamp
        cdef vector[double] denoms = vector[double](self.n_bins, 0.0)
        cdef vector[double] a_data = vector[double]()
        cdef vector[double] b_data = vector[double]()
        cdef double max_b
        cdef int i, j
        cdef double tau
        cdef int count_a, count_b

        for a in self.data['a']:
            a_data.push_back(a)
        for b in self.data['b']:
            b_data.push_back(b)

        max_b = max(b_data) if not b_data.empty() else 0.0

        for i in range(self.n_bins):
            tau = self.bin_edges[i+1]
            count_a = sum(1 for a in a_data if a >= tau)
            count_b = sum(1 for b in b_data if b <= (max_b - tau))
            denoms[i] = count_a * count_b

        return [
            (duration - self.bin_edges[i+1]) / denoms[i] if denoms[i] != 0 else 0
            for i in range(self.n_bins)
        ]

    @property
    def g2(self):
        return [self.G2[i] * self.normalization[i] for i in range(self.n_bins)]

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
                
                for j in range(self.bin[i].size()):
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


#####
# cython: language_level=3
# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False

from libc.stdlib cimport malloc, free
from libc.math cimport fmax, fmin

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
    
    return result, 
