# cython: language_level=3
# distutils: language=c++

from libc.stdio cimport FILE, fopen, fread, fseek, fclose, SEEK_SET, SEEK_END, ftell, perror, SEEK_CUR
from libc.string cimport memcmp, memset
from libc.stdint cimport int32_t, int64_t, uint32_t

from libc.stdlib cimport malloc, free, realloc
from libc.stdio cimport fopen, fread, fclose, FILE
from libcpp.map cimport map

import time
import sys
import struct
import io

TIME_OVERFLOW = 210698240
T3_WRAPAROUND = 2 ** 16


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


def t3_to_histogram(str path, double min_ns = -1, double max_ns = 0xFFFFFFFF):
    tags, header_end = read_header(path)
    num_records = tags["TTResult_NumberOfRecords"]
    resolution = tags["MeasDesc_Resolution"] * 1e9  # in nanoseconds

    if min_ns > max_ns:
        raise ValueError("min_dtime cannot be larger than max_dtime.")

    cdef FILE *fp = fopen(path.encode('utf-8'), "rb")
    fseek(fp, header_end, SEEK_SET)

    bin_size_ps = tags["MeasDesc_Resolution"] * 1e12
    sync_rate_Hz = tags["TTResult_SyncRate"]
    histogrammer = T3Histogrammer(
        bin_size_ps=bin_size_ps,
        sync_rate_Hz=sync_rate_Hz,
        min_ns=min_ns,
        max_ns=max_ns,
    )

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
            histogrammer.click(record)
            
    fclose(fp)
    return histogrammer.times_ns, histogrammer.counts


cdef class T3Histogrammer:
    cdef public double bin_size_ns
    cdef public double min_bin, max_bin
    cdef public unsigned int overflow_correction
    cdef uint32_t *counts_arr
    cdef uint32_t *new_counts_arr
    cdef public int array_size
    cdef int new_size

    def __cinit__(self, bin_size_ps, sync_rate_Hz, double min_ns=-1, double max_ns=0xFFFFFFFF):
        self.bin_size_ns = bin_size_ps / 1000  # change to nanoseconds
        # convert max/min from nanoseconds to bin number
        self.min_bin, self.max_bin = min_ns * 1000 / bin_size_ps, max_ns * 1000 / bin_size_ps
        self.overflow_correction = 0
        # Determine the size of the counts array based on sync_rate
        # The latest timestamp we should get is equal to the sync period
        self.array_size = int(1e9 / (sync_rate_Hz * self.bin_size_ns))
        self.counts_arr = <uint32_t*>malloc(self.array_size * sizeof(uint32_t))
        if self.counts_arr is NULL:
            raise MemoryError("Failed to allocate memory for counts array.")
        # Initialize array with zeros
        memset(self.counts_arr, 0, self.array_size * sizeof(uint32_t))

    property times_ns:
        def __get__(self):
            return [i * self.bin_size_ns for i in range(self.array_size) if (self.min_bin <= i) and (i <= self.max_bin)]

    property counts:
        def __get__(self):
            return [self.counts_arr[i] for i in range(self.array_size) if (self.min_bin <= i) and (i <= self.max_bin)]

    def click(self, uint32_t record):
        cdef uint32_t dtime = (record & 0x0FFF0000) >> 16
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
