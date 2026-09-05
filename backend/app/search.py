"""Turning a retrieved chunk into something worth reading in a list.

A chunk is five hundred tokens — most of a screen — and a page of twenty of
them is not a result list, it is the corpus with extra steps. So a result shows
the part of the chunk the query actually landed on, with the matching words
marked.

None of this decides *which* results come back: the index has already ranked
them, and this only chooses where to cut and what to embolden. That is what
lets the matching be approximate, which in Russian it has to be — the index
found «вишлисте» for a query of «вишлист», and a highlighter that insisted on
the exact string would mark nothing at all on a result that is perfectly good.
"""

import re

#: Characters of context in a result. Two lines or so: enough to see why the
#: fragment matched, short enough that twenty of them stay scannable.
WIDTH = 260

#: How much of the match to keep on its left, so the marked word does not sit
#: flush against the ellipsis.
_LEAD = 70

#: Words shorter than this are not marked. It costs the odd initialism, and it
#: buys not painting «что», «как» and «для» yellow on every single result.
_MIN_WORD = 4

#: Match on this much of a word. Russian inflects the ending and almost never
#: the first six characters, so a prefix is a stemmer's worth of behaviour for
#: none of its cost — and six is long enough that «дом» cannot claim «документ».
_STEM = 6

#: A ceiling on the marking, not on the search. A one-word query against a
#: chunk that repeats that word forty times would otherwise build forty runs.
_MAX_MARKS = 20

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def stems(query: str) -> list[str]:
    """The parts of the query worth matching on, lowercased."""
    seen: list[str] = []
    for word in _WORD_RE.findall(query.lower()):
        stem = word[:_STEM]
        if len(word) >= _MIN_WORD and stem not in seen:
            seen.append(stem)
    return seen


def snippet(text: str, query: str) -> list[tuple[str, bool]]:
    """The part of ``text`` worth showing, as ``(run, is_match)`` pairs.

    Concatenating the runs gives exactly the visible snippet, ellipses and all,
    so the caller renders them in order and needs to know nothing else.
    """
    body = " ".join(text.split())
    marks = _marks(body, stems(query))
    start, end = _window(body, marks)
    visible = [(a, b) for a, b in marks if a >= start and b <= end]

    runs: list[tuple[str, bool]] = []
    if start > 0:
        runs.append(("…", False))
    cursor = start
    for a, b in visible:
        if a > cursor:
            runs.append((body[cursor:a], False))
        runs.append((body[a:b], True))
        cursor = b
    if cursor < end:
        runs.append((body[cursor:end], False))
    if end < len(body):
        runs.append(("…", False))
    return [run for run in runs if run[0]]


def _marks(body: str, wanted: list[str]) -> list[tuple[int, int]]:
    """Where the query words appear, merged and in order.

    Each hit is widened to the end of the word it started, so a query for
    «вишлист» marks the whole of «вишлисте» rather than stopping mid-word.
    """
    lowered = body.lower()
    found: list[tuple[int, int]] = []
    for stem in wanted:
        at = lowered.find(stem)
        while at != -1 and len(found) < _MAX_MARKS:
            end = at + len(stem)
            while end < len(body) and body[end].isalnum():
                end += 1
            found.append((at, end))
            at = lowered.find(stem, end)

    merged: list[tuple[int, int]] = []
    for start, end in sorted(found):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _window(body: str, marks: list[tuple[int, int]]) -> tuple[int, int]:
    """Which slice of the body to show: around the first match, or the head.

    Cut on spaces rather than mid-word — a snippet that begins «…исление НДС»
    reads as a bug even when the text is right.
    """
    if len(body) <= WIDTH:
        return 0, len(body)
    start = max(0, marks[0][0] - _LEAD) if marks else 0
    if start:
        space = body.find(" ", start)
        start = space + 1 if 0 <= space < start + 20 else start
    end = min(len(body), start + WIDTH)
    if end < len(body):
        space = body.rfind(" ", start, end)
        end = space if space > start + WIDTH // 2 else end
    return start, end
