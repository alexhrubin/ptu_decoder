import numpy as np
import pytest
from pathlib import Path

from ptu_decoder import (
    read_header,
    t2_to_timestamps,
    t3_to_histogram,
    t2_iterator,
    T2Streamer,
    T3Histogrammer,
    pcorrelate,
)

DATA = Path(__file__).parent.parent / "example_ptu_files"
T2_FILE_1 = str(DATA / "T2_file_1.ptu")
T2_FILE_2 = str(DATA / "T2_file_2.ptu")
T3_FILE_1 = str(DATA / "T3_file_1.ptu")
T3_FILE_2 = str(DATA / "T3_file_2.ptu")


# ---------------------------------------------------------------------------
# read_header
# ---------------------------------------------------------------------------

class TestReadHeader:
    def test_returns_dict_and_int(self):
        tags, end = read_header(T2_FILE_1)
        assert isinstance(tags, dict)
        assert isinstance(end, int)
        assert end > 0

    def test_required_keys_present(self):
        tags, _ = read_header(T2_FILE_1)
        for key in ("TTResult_NumberOfRecords", "MeasDesc_Resolution",
                    "MeasDesc_GlobalResolution", "TTResultFormat_TTTRRecType"):
            assert key in tags, f"Missing tag: {key}"

    def test_invalid_file_raises(self, tmp_path):
        bad = tmp_path / "bad.ptu"
        bad.write_bytes(b"NOTPQTTTR" + b"\x00" * 100)
        with pytest.raises(ValueError, match="Magic invalid"):
            read_header(str(bad))

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_header("nonexistent.ptu")


# ---------------------------------------------------------------------------
# t2_to_timestamps
# ---------------------------------------------------------------------------

class TestT2ToTimestamps:
    def test_returns_numpy_arrays(self):
        ch0, ch1 = t2_to_timestamps(T2_FILE_1)
        assert isinstance(ch0, np.ndarray)
        assert isinstance(ch1, np.ndarray)

    def test_event_counts(self):
        ch0, ch1 = t2_to_timestamps(T2_FILE_1)
        assert len(ch0) == 34663
        assert len(ch1) == 987

    def test_timestamps_sorted(self):
        ch0, ch1 = t2_to_timestamps(T2_FILE_1)
        assert np.all(np.diff(ch0) >= 0), "ch0 timestamps not monotonically increasing"
        assert np.all(np.diff(ch1) >= 0), "ch1 timestamps not monotonically increasing"

    def test_timestamps_positive(self):
        ch0, ch1 = t2_to_timestamps(T2_FILE_1)
        assert ch0[0] > 0
        assert ch1[0] > 0

    def test_first_last_values(self):
        ch0, ch1 = t2_to_timestamps(T2_FILE_1)
        assert ch0[0]  == pytest.approx(839383.556,    rel=1e-6)
        assert ch0[-1] == pytest.approx(9999776594.68, rel=1e-6)
        assert ch1[0]  == pytest.approx(55903960.16,   rel=1e-6)

    def test_file2_event_counts(self):
        ch0, ch1 = t2_to_timestamps(T2_FILE_2)
        assert len(ch0) == 77562
        assert len(ch1) == 1279783

    def test_wrong_record_type_raises(self):
        with pytest.raises(ValueError, match="PicoHarp T2"):
            t2_to_timestamps(T3_FILE_1)


# ---------------------------------------------------------------------------
# t2_iterator
# ---------------------------------------------------------------------------

class TestT2Iterator:
    def test_yields_tuples(self):
        it = t2_iterator(T2_FILE_1)
        ch, ts = next(it)
        assert isinstance(ch, int)
        assert isinstance(ts, float)

    def test_matches_t2_to_timestamps(self):
        ch0_bulk, ch1_bulk = t2_to_timestamps(T2_FILE_1)

        events = list(t2_iterator(T2_FILE_1))
        iter_ch0 = np.array([ts for ch, ts in events if ch == 0])
        iter_ch1 = np.array([ts for ch, ts in events if ch == 1])

        assert len(iter_ch0) == len(ch0_bulk)
        assert len(iter_ch1) == len(ch1_bulk)
        assert np.allclose(iter_ch0, ch0_bulk)
        assert np.allclose(iter_ch1, ch1_bulk)

    def test_stops_at_eof(self):
        events = list(t2_iterator(T2_FILE_1))
        assert len(events) > 0  # consumed without error

    def test_wrong_record_type_raises(self):
        with pytest.raises(ValueError, match="PicoHarp T2"):
            t2_iterator(T3_FILE_1)


# ---------------------------------------------------------------------------
# t3_to_histogram
# ---------------------------------------------------------------------------

class TestT3ToHistogram:
    def test_returns_numpy_arrays(self):
        times, counts = t3_to_histogram(T3_FILE_1)
        assert isinstance(times, np.ndarray)
        assert isinstance(counts, np.ndarray)

    def test_same_length(self):
        times, counts = t3_to_histogram(T3_FILE_1)
        assert len(times) == len(counts)

    def test_bin_count(self):
        times, counts = t3_to_histogram(T3_FILE_1)
        assert len(times) == 1562

    def test_total_counts(self):
        times, counts = t3_to_histogram(T3_FILE_1)
        assert int(np.sum(counts)) == 330651

    def test_times_uniformly_spaced(self):
        times, counts = t3_to_histogram(T3_FILE_1)
        diffs = np.diff(times)
        assert np.allclose(diffs, diffs[0]), "bin spacing is not uniform"

    def test_times_start_at_zero(self):
        times, _ = t3_to_histogram(T3_FILE_1)
        assert times[0] == pytest.approx(0.0)

    def test_peak_location(self):
        times, counts = t3_to_histogram(T3_FILE_1)
        peak_idx = int(np.argmax(counts))
        assert times[peak_idx] == pytest.approx(35.904, rel=1e-4)
        assert int(counts[peak_idx]) == 13097

    def test_file2_total_counts(self):
        times, counts = t3_to_histogram(T3_FILE_2)
        assert int(np.sum(counts)) == 570655

    def test_min_max_filter(self):
        times, counts = t3_to_histogram(T3_FILE_1, min_ns=1.0, max_ns=5.0)
        assert times[0]  == pytest.approx(0.96,  rel=1e-4)
        assert times[-1] == pytest.approx(4.992, rel=1e-4)
        assert int(np.sum(counts)) == 337

    def test_min_max_filter_no_counts_outside_range(self):
        times, counts = t3_to_histogram(T3_FILE_1, min_ns=1.0, max_ns=5.0)
        assert times[0]  >= 1.0 - 0.1  # allow one bin width of slack
        assert times[-1] <= 5.0 + 0.1

    def test_segments_total_equals_unsegmented(self):
        _, counts_all = t3_to_histogram(T3_FILE_1)
        hists = t3_to_histogram(T3_FILE_1, segments=3)
        total_segmented = sum(int(np.sum(h.counts)) for h in hists)
        assert total_segmented == int(np.sum(counts_all))

    def test_segments_returns_list(self):
        result = t3_to_histogram(T3_FILE_1, segments=3)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_segments_counts(self):
        hists = t3_to_histogram(T3_FILE_1, segments=3)
        assert int(np.sum(hists[0].counts)) == 132992
        assert int(np.sum(hists[1].counts)) == 109350
        assert int(np.sum(hists[2].counts)) == 88309

    def test_single_segment_returns_tuple(self):
        result = t3_to_histogram(T3_FILE_1, segments=1)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_wrong_record_type_raises(self):
        with pytest.raises(ValueError, match="PicoHarp T3"):
            t3_to_histogram(T2_FILE_1)

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            t3_to_histogram(T3_FILE_1, segments=0)
        with pytest.raises(ValueError):
            t3_to_histogram(T3_FILE_1, min_ns=5.0, max_ns=1.0)
        with pytest.raises(ValueError):
            t3_to_histogram(T3_FILE_1, segments=2, window_ends_sec=[1.0])


# ---------------------------------------------------------------------------
# T2Streamer
# ---------------------------------------------------------------------------

class TestT2Streamer:
    def test_click_records_photon(self):
        s = T2Streamer(resolution_ps=4000)
        # channel=0, time=1000 ticks
        record = (0 << 28) | 1000
        s.click(record)
        assert s.ch0_count == 1
        assert s.ch0_times_ns[0] == pytest.approx(1000 * 4.0)

    def test_overflow_accumulates(self):
        s = T2Streamer(resolution_ps=4000)
        # overflow record: channel=0xF, markers=0
        overflow_record = (0xF << 28)
        s.click(overflow_record)
        # photon at time=0 after one overflow
        record = (0 << 28) | 0
        s.click(record)
        assert s.ch0_times_ns[0] == pytest.approx(210698240 * 4.0)

    def test_clear_resets_state(self):
        s = T2Streamer(resolution_ps=4000)
        record = (0 << 28) | 500
        s.click(record)
        assert s.ch0_count == 1
        s.clear()
        assert s.ch0_count == 0
        assert s.ch1_count == 0
        assert len(s.ch0_times_ns) == 0
        assert len(s.ch1_times_ns) == 0

    def test_batch_matches_individual_clicks(self):
        records = [(0 << 28) | i for i in range(100)]
        s1 = T2Streamer(resolution_ps=4000)
        for r in records:
            s1.click(r)
        s2 = T2Streamer(resolution_ps=4000)
        s2.batch(records)
        assert np.allclose(s1.ch0_times_ns, s2.ch0_times_ns)

    def test_channel_separation(self):
        s = T2Streamer(resolution_ps=4000)
        s.click((0 << 28) | 100)
        s.click((1 << 28) | 200)
        assert s.ch0_count == 1
        assert s.ch1_count == 1
        assert s.ch0_times_ns[0] == pytest.approx(100 * 4.0)
        assert s.ch1_times_ns[0] == pytest.approx(200 * 4.0)

    def test_returns_numpy_arrays(self):
        s = T2Streamer(resolution_ps=4000)
        s.click((0 << 28) | 10)
        assert isinstance(s.ch0_times_ns, np.ndarray)
        assert isinstance(s.ch1_times_ns, np.ndarray)

    def test_marker_record_ignored(self):
        s = T2Streamer(resolution_ps=4000)
        # channel=0xF, markers != 0 → marker, not overflow
        marker_record = (0xF << 28) | 0b0001
        s.click(marker_record)
        assert s.ch0_count == 0
        assert s.ch1_count == 0


# ---------------------------------------------------------------------------
# T3Histogrammer
# ---------------------------------------------------------------------------

class TestT3Histogrammer:
    def test_click_records_photon(self):
        h = T3Histogrammer(bin_size_ps=12000, sync_rate_Hz=80_000_000)
        h._click(10)
        assert h.counts[10] == 1

    def test_overflow_record_ignored(self):
        h = T3Histogrammer(bin_size_ps=12000, sync_rate_Hz=80_000_000)
        # overflow record: channel=0xF0000000, dtime=0
        overflow = 0xF0000000
        h.click(overflow)
        assert int(np.sum(h.counts)) == 0

    def test_marker_record_not_counted(self):
        h = T3Histogrammer(bin_size_ps=12000, sync_rate_Hz=80_000_000)
        # marker: channel=0xF0000000, dtime!=0
        marker = 0xF0000000 | (5 << 16) | 1
        h.click(marker)
        assert int(np.sum(h.counts)) == 0

    def test_clear_resets_counts(self):
        h = T3Histogrammer(bin_size_ps=12000, sync_rate_Hz=80_000_000)
        h._click(50)
        h._click(50)
        assert h.counts[50] == 2
        h.clear()
        assert int(np.sum(h.counts)) == 0

    def test_times_ns_spacing(self):
        # bin_size_ps=100 (0.1 ns/bin), 80 MHz sync → array_size=125
        h = T3Histogrammer(bin_size_ps=100, sync_rate_Hz=80_000_000)
        times = h.times_ns
        assert times[0] == pytest.approx(0.0)
        assert times[1] == pytest.approx(0.1)   # 100 ps = 0.1 ns

    def test_min_max_filter(self):
        # bin_size_ps=100 → min_bin=10, max_bin=20
        h = T3Histogrammer(bin_size_ps=100, sync_rate_Hz=80_000_000,
                           min_ns=1.0, max_ns=2.0)
        h._click(0)   # below min_bin=10 → excluded
        h._click(15)  # in range (15 * 0.1 ns = 1.5 ns) → included
        h._click(30)  # above max_bin=20 → excluded
        assert int(np.sum(h.counts)) == 1

    def test_returns_numpy_arrays(self):
        h = T3Histogrammer(bin_size_ps=12000, sync_rate_Hz=80_000_000)
        assert isinstance(h.times_ns, np.ndarray)
        assert isinstance(h.counts, np.ndarray)

    def test_times_and_counts_same_length(self):
        h = T3Histogrammer(bin_size_ps=12000, sync_rate_Hz=80_000_000)
        assert len(h.times_ns) == len(h.counts)

    def test_batch_matches_individual(self):
        dtimes = list(range(50, 150))
        h1 = T3Histogrammer(bin_size_ps=12000, sync_rate_Hz=80_000_000)
        for d in dtimes:
            h1._click(d)
        h2 = T3Histogrammer(bin_size_ps=12000, sync_rate_Hz=80_000_000)
        records = [(1 << 28) | (d << 16) | 1 for d in dtimes]
        h2.batch(records)
        assert np.array_equal(h1.counts, h2.counts)


# ---------------------------------------------------------------------------
# pcorrelate
# ---------------------------------------------------------------------------

class TestPcorrelate:
    def test_returns_list(self):
        result = pcorrelate([1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [0.0, 0.5, 1.0])
        assert isinstance(result, list)

    def test_output_length(self):
        bins = [0.0, 1.0, 2.0, 3.0]
        result = pcorrelate([1.0, 2.0], [1.5, 2.5], bins)
        assert len(result) == len(bins) - 1

    def test_identical_streams_positive(self):
        # Autocorrelation at short lag should be positive
        t = list(np.arange(0, 100, 1.0))
        result = pcorrelate(t, t, [0.5, 1.5, 2.5, 3.5])
        assert all(v >= 0 for v in result)

    def test_zero_lag_bin_highest(self):
        # At the smallest lag, coincidences should be highest
        t = list(np.arange(0, 100, 1.0))
        bins = [0.0, 1.0, 2.0, 4.0, 8.0]
        result = pcorrelate(t, t, bins)
        assert result[0] == max(result)

    def test_empty_streams(self):
        result = pcorrelate([], [], [0.0, 1.0, 2.0])
        assert result == [0.0, 0.0]

    def test_normalize_flag(self):
        t = list(np.arange(0, 50, 1.0))
        u = list(np.arange(0, 50, 1.0))
        unnorm = pcorrelate(t, u, [0.0, 1.0, 2.0])
        normed = pcorrelate(t, u, [0.0, 1.0, 2.0], normalize=True)
        assert unnorm != normed
