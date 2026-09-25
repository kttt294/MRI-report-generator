from scripts.v2_2_bounded_decode_validation import repeated_fivegram_fraction


def test_repeated_fivegrams_detect_generation_loop():
    unique = "alpha beta gamma delta epsilon zeta eta theta"
    loop = "alpha beta gamma delta epsilon " * 8
    assert repeated_fivegram_fraction(unique) == 0
    assert repeated_fivegram_fraction(loop) > 0.7
