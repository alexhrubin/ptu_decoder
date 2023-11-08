# cython: language_level=3

from libc.stdio cimport FILE, fopen, fread, fseek, fclose, SEEK_SET, SEEK_END, ftell, perror, SEEK_CUR
from libc.string cimport memcmp
from libc.stdint cimport int32_t, int64_t, uint32_t

from libc.stdlib cimport malloc, free
from libc.stdio cimport fopen, fread, fclose, FILE
from libcpp.map cimport map

import time
import sys
import struct
import io

TIME_OVERFLOW = 210698240


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


def t3_to_histogram(str path, int min_dtime = -1, int max_dtime = -1):
    tags, header_end = read_header(path)
    num_records = tags["TTResult_NumberOfRecords"]
    resolution = tags["MeasDesc_Resolution"] * 1e9  # in nanoseconds

    if min_dtime > max_dtime:
        raise ValueError("min_dtime cannot be larger than max_dtime.")

    return __t3_to_histogram(path, header_end, num_records, resolution, min_dtime, max_dtime)


cdef dict __t3_to_histogram(str path, int header_end, int num_records, float resolution, int min_dtime = -1, int max_dtime = -1):
    cdef:
        FILE *fp = fopen(path.encode('utf-8'), "rb")
        unsigned int T3_WRAPAROUND = 2 ** 16
        unsigned int overflow = 0
        unsigned int record
        int i
        int dtime
        map[float, int] counts_map
        char *buffer
        size_t record_size = sizeof(record)
        size_t buffer_size = record_size * 10000  # adjust the buffer size to your needs
        size_t read_count

    if not fp:
        raise FileNotFoundError("Could not open file: " + path)

    fseek(fp, header_end, SEEK_SET)

    # Allocate a buffer to read in bulk
    buffer = <char *>malloc(buffer_size)
    if not buffer:
        raise MemoryError("Failed to allocate buffer")

    try:
        while True:
            # Read in a buffer's worth of data
            read_count = fread(buffer, record_size, 10000, fp)
            if read_count == 0:
                break

            # Process each record in the buffer
            for i in range(read_count):
                record = (<unsigned int *>buffer)[i]
                dtime = (record & 0x0FFF0000) >> 16

                dtime_ns = dtime * resolution
                if (min_dtime != -1 and dtime_ns < min_dtime) or (max_dtime != -1 and dtime_ns > max_dtime):
                    continue

                counts_map[dtime_ns] += 1
    finally:
        free(buffer)
        fclose(fp)

    # Convert the C++ map to a Python dict
    counts = dict(counts_map)

    return counts



cdef class T3Histogrammer:
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

    def histogram(self, num_records, min=None, max=None):
        fseek(self.fp, 3584 + 48, SEEK_SET); # go to the start of the timing data

        cdef unsigned int T3_WRAPAROUND = 2 ** 16  # 65536
        cdef unsigned int overflow = 0
        cdef unsigned int record

        counts = {}

        for _ in range(num_records):
            fread(&record, sizeof(record), 1, self.fp)

            dtime = (record & 0x0FFF0000) >> 16;

            if (min and dtime < min) or (max and dtime > max):
                continue
            
            if dtime in counts:
                counts[dtime] += 1
            else:
                counts[dtime] = 1

        return counts





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
