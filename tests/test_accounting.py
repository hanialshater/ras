from ras.accounting import METHOD_FOOTPRINTS, memory_rows


def test_binary_joint_memory_5m_100k():
    binary = METHOD_FOOTPRINTS["binary1_ls2_int4"]
    assert binary.total_bytes(5_000_000, 100_000) == 301_600_000


def test_pq_program_store_is_much_larger_than_binary():
    binary = METHOD_FOOTPRINTS["binary1_ls2_int4"]
    pq = METHOD_FOOTPRINTS["pq64_linear_lut"]
    assert pq.program_bytes / binary.program_bytes > 300


def test_memory_rows_contains_expected_payloads():
    rows = {row["method"]: row for row in memory_rows(5_000_000, 100_000)}
    assert rows["Binary1-LS2-int4"]["total_payload_MB"] == 301.6
    assert rows["PQ64 compiled linear"]["program_payload_MB"] == 6554.8
