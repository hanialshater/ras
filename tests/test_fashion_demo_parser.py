from demos.fashion_app import parse_latents


def test_demo_parses_positive_and_negative_latents():
    pos, neg = parse_latents("minimalist black office shoes not sporty")
    assert "minimalist" in pos
    assert "office_appropriate" in pos
    assert "technical_sporty" in neg
    assert "technical_sporty" not in pos


def test_demo_parses_quiet_luxury():
    pos, neg = parse_latents("quiet luxury accessories")
    assert "quiet_luxury" in pos
    assert neg == []
