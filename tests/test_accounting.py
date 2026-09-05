from ras.accounting import METHOD_FOOTPRINTS, PQ64_SHARED_CODEBOOK_BYTES, memory_rows


def test_binary_joint_memory_5m_100k():
    binary = METHOD_FOOTPRINTS["binary1_ls2_int4"]
    assert binary.total_bytes(5_000_000, 100_000) == 301_600_000


def test_pq64_distinguishes_persistent_head_from_active_lut():
    binary = METHOD_FOOTPRINTS["binary1_ls2_int4"]
    pq = METHOD_FOOTPRINTS["pq64_linear_lut"]
    assert pq.item_bytes == 64
    assert pq.program_bytes == 1_548
    assert pq.active_bytes == 65_548
    assert pq.shared_bytes == 393_216
    assert PQ64_SHARED_CODEBOOK_BYTES == 393_216
    assert 7.0 < pq.program_bytes / binary.program_bytes < 7.3


def test_memory_rows_contains_persistent_and_active_payloads():
    rows = {row["method"]: row for row in memory_rows(5_000_000, 100_000)}
    binary = rows["Binary1-LS2-int4"]
    pq = rows["PQ64 compiled linear"]

    assert binary["total_payload_MB"] == 301.6
    assert abs(float(pq["total_payload_MB"]) - 475.193216) < 1e-9
    assert pq["program_payload_MB"] == 154.8
    assert pq["shared_payload_MB"] == 0.393216
    assert pq["persistent_predicate_B"] == 1_548
    assert pq["active_predicate_B"] == 65_548
