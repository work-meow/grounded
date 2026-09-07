"""The bill: what it adds up, and what it refuses to guess.

The failure this guards is not a crash. It is a total that looks like a total
and is not one — a call whose price the provider did not report, silently
counted as free, in a number somebody bills a customer from.
"""

from app.spend import ANSWER, RERANK, WEB_SEARCH, Spend


def completion(cost, model="google/gemini-2.5-flash-lite", prompt=900, out=12):
    return {
        "model": model,
        "usage": {"prompt_tokens": prompt, "completion_tokens": out, "cost": cost},
    }


def test_a_turn_adds_up_by_stage():
    spend = Spend()
    spend.add(ANSWER, "openai/gpt-5-mini", prompt_tokens=4100, completion_tokens=48, cost_usd=4e-4)
    spend.add(ANSWER, "openai/gpt-5-mini", prompt_tokens=900, completion_tokens=20, cost_usd=1e-4)
    spend.add_completion(RERANK, "google/gemini-2.5-flash-lite", completion(1.7e-4))

    report = spend.report()

    assert report["calls"] == 3
    assert report["prompt_tokens"] == 4100 + 900 + 900
    assert report["total_tokens"] == 5900 + 80
    assert report["cost_usd"] == 0.00067
    assert report["cost_complete"] is True
    # Two calls to the same model in the same stage are one line, three stages
    # would be three: the breakdown answers "what did this go on", not "what
    # happened when".
    assert [(row["stage"], row["calls"]) for row in report["stages"]] == [(ANSWER, 2), (RERANK, 1)]


def test_a_price_nobody_reported_is_missing_rather_than_zero():
    spend = Spend()
    spend.add(ANSWER, "openai/gpt-5-mini", prompt_tokens=100, completion_tokens=10, cost_usd=1e-4)
    spend.add_completion(WEB_SEARCH, "google/gemini-2.5-flash-lite", completion(None))

    report = spend.report()

    # The tokens are known and counted; the total is now a floor, and says so.
    assert report["prompt_tokens"] == 1000
    assert report["cost_usd"] == 0.0001
    assert report["cost_complete"] is False


def test_a_cent_and_a_half_of_a_thousandth_still_shows():
    """Rounding is to eight places for a reason: the judge behind one search
    costs $0.00017, and two decimals fewer would report it as free."""
    spend = Spend()
    spend.add_completion(RERANK, "m", completion(1.7e-4))

    assert spend.report()["cost_usd"] == 0.00017


def test_the_model_that_ran_is_the_one_that_is_billed():
    """OpenRouter is free to route a request to a dated variant, and the bill
    belongs to whatever answered — which is what the response says."""
    spend = Spend()
    spend.add_completion(RERANK, "google/gemini-2.5-flash-lite", completion(1e-6, model="x/y:free"))

    assert spend.report()["stages"][0]["model"] == "x/y:free"


def test_a_body_in_a_shape_this_build_does_not_know_costs_one_call_and_no_answer():
    spend = Spend()
    for body in (None, {}, {"usage": None}, {"usage": []}, "nope"):
        spend.add_completion(ANSWER, "m", body)

    assert spend.report()["calls"] == 0
    assert spend.report()["cost_complete"] is True


def test_numbers_that_are_not_numbers_do_not_reach_the_report():
    spend = Spend()
    spend.add(ANSWER, "m", prompt_tokens="900", completion_tokens=-5, cost_usd="0.001")
    spend.add(ANSWER, "m", prompt_tokens=None, completion_tokens=None, cost_usd=True)

    report = spend.report()
    assert report["total_tokens"] == 0
    assert report["cost_usd"] == 0.0
    # Both calls happened and neither reported a usable price.
    assert report["calls"] == 2
    assert report["cost_complete"] is False


def test_nothing_spent_is_a_complete_report_of_nothing():
    assert Spend().report() == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "cost_complete": True,
        "calls": 0,
        "stages": [],
    }
