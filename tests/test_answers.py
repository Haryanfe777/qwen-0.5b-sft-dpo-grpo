import pytest

from posttrain.answers import (arithmetic_errors, clean_rationale, correctness_reward, extract_lenient,
                               extract_strict, format_reward, gold_answer, is_correct, normalize_number,
                               truncate_completion)

GSM = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips.\n#### 72"


@pytest.mark.parametrize("text,expected", [
    ("1,234", 1234.0), ("$5.50", 5.5), ("72.", 72.0), ("-3", -3.0), ("abc", None), (None, None),
])
def test_normalize_number(text, expected):
    assert normalize_number(text) == expected


def test_gold_and_clean_rationale():
    assert gold_answer(GSM) == 72.0
    cleaned = clean_rationale(GSM)
    assert "<<" not in cleaned and cleaned.endswith("#### 72")
    assert "48/2 = 24 clips" in cleaned


def test_strict_vs_lenient():
    assert extract_strict("so 5 + 3 = 8\n#### 8") == 8.0
    assert extract_strict("the answer is 8") is None
    assert extract_lenient("the answer is 8") == 8.0
    assert extract_strict("#### $1,200") == 1200.0
    assert extract_lenient("no numbers here") is None
    # base models state the answer and then ramble on with more numbers
    assert extract_lenient("so 1 * 2 = 2. The answer is 2.\nYou are an AI assistant 3 4 5") == 2.0


def test_is_correct_tolerance():
    assert is_correct(72.0, 72.0)
    assert is_correct(0.3333333, 0.33333334)
    assert not is_correct(None, 1.0)
    assert not is_correct(71.0, 72.0)


def test_truncate_completion_stops_at_next_question():
    assert truncate_completion("2+2=4\n#### 4\nQuestion: next one") == "2+2=4\n#### 4"


def test_arithmetic_errors():
    assert arithmetic_errors("48 + 24 = 72 then 3 * 4 = 12") == []
    errs = arithmetic_errors("48 + 24 = 70")
    assert len(errs) == 1 and "should be 72" in errs[0]


@pytest.mark.parametrize("line", [
    "5 + 3 + 2 = 10",           # chained: only the last two operands used to be checked
    "2 + 3 * 4 = 14",           # operator precedence
    "50 * 20 / 100 = 10",
    "(2 + 3) * 4 = 20",         # brackets
    "-3 + 5 = 2",               # leading negative
    "$1,200 / 4 = $300",        # currency and thousands separators
    "3 x 4 = 12 and 12 ÷ 4 = 3",
    "10 / 3 = 3.33",            # rounded result is within tolerance
    # not checkable, so skipped rather than misreported (all from real GSM8K solutions):
    "There are 60 students x 2/3 = 40 students",   # match would start mid-expression
    "8 x 1 1/2 = 12 dinner rolls",                  # mixed number on the left
    "1/2 * 1/2 = 1/4",                              # fraction on the right
    "3 / 4 * 2 = 1 1/2 square feet",                # mixed number on the right
    "192 / 1.2 = 1.2Y / 1.2",                       # algebra
    "4080 + 4080 / 2 = 4000 + 2040 = 6120",         # chain continues after '='
    "Max 5 + 3 = 8",                                # a word ending in 'x' is not an operator
])
def test_arithmetic_errors_accepts_correct_lines(line):
    assert arithmetic_errors(line) == []


def test_arithmetic_errors_flags_wrong_chain():
    errs = arithmetic_errors("5 + 3 + 2 = 11")
    assert len(errs) == 1 and "should be 10" in errs[0]
    # a real slip in a GSM8K reference solution
    assert len(arithmetic_errors("So it cost 192*100=$1920")) == 1


def test_rewards():
    comps = ["2+2=4\n#### 4", "#### 5", "the answer is 4", "#### 4\n#### 4"]
    gold = [4.0, 4.0, 4.0, 4.0]
    assert correctness_reward(comps, gold=gold) == [1.0, 0.0, 0.0, 1.0]
    assert format_reward(comps) == [1.0, 1.0, 0.0, 0.0]
    # conversational format is accepted too
    assert correctness_reward([[{"role": "assistant", "content": "#### 4"}]], gold=[4.0]) == [1.0]