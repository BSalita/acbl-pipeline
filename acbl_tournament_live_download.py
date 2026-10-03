"""
Download ACBL tournament results from live.acbl.org when the API PAT is rejected.

Uses the same persistent real-Chrome session as club downloading
(``acbl_club_download_to_json.fetch_acbl_page``), which waits through
Cloudflare and the AWS WAF interstitial in front of live.acbl.org.

Sanction files keep the event shape ``build_session_ids_from_sanctioned_events``
already reads (``id`` + ``session_count``). Session files keep the API JSON
shape consumed by ``acbl_tournament_sessions_json_to_sql``: ``handrecord`` and
``sections[].board_results``.
"""

from __future__ import annotations

import html as html_lib
import json
import re
import time
import urllib.parse
from datetime import datetime
from typing import Any, Callable

import pathlib

from acbl_club_download_to_json import (
    Forbidden403Error,
    fetch_acbl_page,
    shutdown_acbl_browser,
)
from acbl_tournament_download_sessions_using_sanctioned_events import DownloadStats

LIVE_ORIGIN = "https://live.acbl.org"
TOURNAMENT_LIST_TYPES = ("nabc", "regionals", "sectionals")

_SUIT_LETTER = {
    "clubs": "C",
    "diams": "D",
    "diamonds": "D",
    "hearts": "H",
    "spades": "S",
}
_SUIT_FIELD = {
    "clubs": "clubs",
    "diams": "diamonds",
    "diamonds": "diamonds",
    "hearts": "hearts",
    "spades": "spades",
}
_DEALER = {
    "N": "North",
    "E": "East",
    "S": "South",
    "W": "West",
    "NORTH": "North",
    "EAST": "East",
    "SOUTH": "South",
    "WEST": "West",
}
_VUL = {
    "NONE": "None",
    "LOVE": "None",
    "-": "None",
    "NS": "N-S",
    "N-S": "N-S",
    "EW": "E-W",
    "E-W": "E-W",
    "BOTH": "Both",
    "ALL": "Both",
}
_RANK = {"a": "A", "k": "K", "q": "Q", "j": "J", "t": "10"}
_EMPTY_SUITS = {
    "spades": "-----",
    "hearts": "-----",
    "diamonds": "-----",
    "clubs": "-----",
}


def shutdown_browser() -> None:
    shutdown_acbl_browser()


def split_session_id(session_id: str) -> tuple[str, str, int]:
    """Return ``(sanction, event_code, session_number)`` for a live session id.

    ``2604346-1001-1`` and ``NABC262-OSHL-1`` both split on the last hyphen
    for the session number, then on the first hyphen of the event id.
    """
    event_id, sep, number = (session_id or "").rpartition("-")
    sanction, code_sep, code = event_id.partition("-")
    if not sep or not code_sep or not number.isdigit() or not sanction or not code:
        raise ValueError(f"not a live.acbl.org session id: {session_id}")
    return sanction, code, int(number)


def _abs_live(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("//"):
        return "https:" + url
    if not url.startswith("/"):
        url = "/" + url
    return LIVE_ORIGIN + url


def _text(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment or "")
    text = html_lib.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _parse_month_date(text: str) -> datetime | None:
    text = text.strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _overlaps(
    tourn_start: datetime | None,
    tourn_end: datetime | None,
    start: datetime | None,
    end: datetime | None,
) -> bool:
    """True when a tournament's dates meet the requested window."""
    if tourn_end is not None and start is not None and tourn_end.date() < start.date():
        return False
    if tourn_start is not None and end is not None and tourn_start.date() > end.date():
        return False
    return True


def _clock(text: str) -> str:
    match = re.search(r"(\d{1,2}):(\d{2})\s*([ap]m)?", text or "", re.I)
    if not match:
        return ""
    hour = int(match.group(1))
    minute = int(match.group(2))
    ampm = (match.group(3) or "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute:02d}"


def _number(text: str) -> int | float | None:
    cleaned = (text or "").strip().replace(",", "")
    if not cleaned or cleaned == "-":
        return None
    try:
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return None


def _acbl_numbers(title: str) -> list[int | str]:
    numbers: list[int | str] = []
    for part in (title or "").split("|"):
        part = part.strip()
        if not part:
            continue
        numbers.append(int(part) if part.isdigit() else part)
    return numbers


def _normalize_suit(raw: str) -> str:
    text = html_lib.unescape(raw or "").replace("\u2014", "").replace("\u2013", "")
    text = re.sub(r"\s+", " ", text).strip()
    if not text or not re.search(r"[AKQJ2-9]", text, re.I):
        return "-----"
    return text


_DD_TOKEN = re.compile(
    r'<div class="reverse">(.*?)</div>'
    r'|(-|\d+)/\s*(\d+)\s*<span class="([a-z]+) symbol"'
    r'|(\d+)\s*<span class="([a-z]+) symbol"'
    r'|(-|\d+)/\s*(\d+)\s*NT'
    r'|(\d+)\s*NT'
    r'|NT(\d+(?:/\d+)?)',
    re.I | re.S,
)
_API_DD_TOKEN = re.compile(
    r"^(?:[CDHS]|NT)\d+(?:/\d+)?$"
    r"|^\d+(?:/\d+)?(?:[CDHS]|NT)$",
    re.I,
)


def _dd_suit(token: str) -> str:
    if re.search(r"NT", token, re.I):
        return "NT"
    match = re.search(r"[CDHS]", token, re.I)
    return match.group(0).upper() if match else ""


def _decode_reverse(inner: str) -> str:
    suit = re.search(r'class="([a-z]+) symbol"', inner)
    text = re.sub(r"\s+", "", _text(inner)).upper()
    if suit:
        return f"{_SUIT_LETTER.get(suit.group(1), '')}{text}"
    if text.startswith("NT"):
        return text
    return ""


def _dd_side(fragment: str) -> str:
    """Five API double-dummy tokens, one per strain, in C D H S NT order.

    Live pages write a make as ``4/ 5<span diamonds>`` (``4/5D``) and a defeat
    inside ``div.reverse``. A leading ``-`` in a slash is level 0 (``-/1H`` is
    ``0/1H``). The first valid token for each strain wins; a later reverse
    token for the same strain is the same result in trick-count form.
    """
    by_suit: dict[str, str] = {}
    for match in _DD_TOKEN.finditer(fragment):
        if match.group(1) is not None:
            token = _decode_reverse(match.group(1))
        elif match.group(4):
            level = "0" if match.group(2) == "-" else match.group(2)
            token = f"{level}/{match.group(3)}{_SUIT_LETTER.get(match.group(4), '')}"
        elif match.group(6):
            token = f"{match.group(5)}{_SUIT_LETTER.get(match.group(6), '')}"
        elif match.group(8):
            level = "0" if match.group(7) == "-" else match.group(7)
            token = f"{level}/{match.group(8)}NT"
        elif match.group(9):
            token = f"{match.group(9)}NT"
        elif match.group(10):
            token = f"NT{match.group(10)}"
        else:
            continue
        token = token.upper()
        if not _API_DD_TOKEN.match(token):
            continue
        suit = _dd_suit(token)
        if suit and suit not in by_suit:
            by_suit[suit] = token
    return " ".join(by_suit[suit] for suit in ("C", "D", "H", "S", "NT") if suit in by_suit)


def _looks_like_par(text: str) -> bool:
    parts = (text or "").split()
    if len(parts) != 2:
        return False
    try:
        int(parts[0])
    except ValueError:
        return False
    body = parts[1].replace("NT", "N")
    for contract in body.split("/"):
        if not re.fullmatch(r"\d[CDHSN]\**-(?:NS|EW|[NSEW])(?:[+-]\d+)?", contract):
            return False
    return True


def _par_from_fragment(fragment: str) -> str:
    frag = re.sub(r'<span class="([a-z]+) symbol"></span>', lambda m: _SUIT_LETTER.get(m.group(1), ""), fragment)
    frag = re.sub(r"<[^>]+>", " ", frag)
    frag = re.sub(r"\s+", " ", html_lib.unescape(frag)).strip()
    frag = re.sub(r"^Par Score\s*", "", frag, flags=re.I)
    match = re.search(r"([+-]?\d+)\s+(\d\S*(?:/\d\S*)*)", frag)
    if not match:
        return ""
    candidate = f"{match.group(1)} {match.group(2)}"
    return candidate if _looks_like_par(candidate) else ""


def _suits_from_block(block: str) -> dict[str, str]:
    found = re.findall(r'<span class="([a-z]+) symbol"></span>\s*([^<]*)', block)
    suits = dict(_EMPTY_SUITS)
    for suit, text in found:
        field = _SUIT_FIELD.get(suit)
        if field:
            suits[field] = _normalize_suit(text)
    return suits


def _bbo_suits(compact: str) -> dict[str, str]:
    suits = dict(_EMPTY_SUITS)
    names = {"s": "spades", "h": "hearts", "d": "diamonds", "c": "clubs"}
    for suit, cards in re.findall(r"([cdsh])([akqjt2-9]*)", (compact or "").lower()):
        pretty: list[str] = []
        for ch in cards:
            pretty.append(_RANK.get(ch, ch))
        if pretty:
            suits[names[suit]] = " ".join(pretty)
    return suits


def _bbo_params(href: str) -> dict[str, str]:
    raw = html_lib.unescape(href or "")
    parsed = urllib.parse.urlparse(raw)
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    return {key: values[0] for key, values in query.items() if values}


def _par_from_bbo(params: dict[str, str]) -> str:
    blob = html_lib.unescape(urllib.parse.unquote(params.get("p") or ""))
    blob = re.sub(r"<[^>]+>", " ", blob).replace("!", "")
    blob = re.sub(r"\s+", " ", blob)
    match = re.search(r"Optimal Contract:\s*(.+?)(?:\s*Double Dummy|$)", blob, re.I)
    if not match:
        return ""
    candidate = match.group(1).strip().strip("{}").strip()
    candidate = re.sub(r"(NT|[CDHS])xx", r"\1**", candidate, flags=re.I)
    candidate = re.sub(r"(NT|[CDHS])x", r"\1*", candidate, flags=re.I)
    return candidate if _looks_like_par(candidate) else ""


def parse_tournament_list(page_html: str) -> tuple[list[dict[str, Any]], int]:
    """Return tournament stubs and the highest pager page number."""
    found: list[dict[str, Any]] = []
    for href, sanction, name, dates in re.findall(
        r'href="(/events/([^"]+))"[^>]*>\s*<b>(.*?)</b>.*?<i>(.*?)</i>',
        page_html,
        flags=re.S,
    ):
        start_text, _, end_text = _text(dates).partition(" - ")
        found.append(
            {
                "sanction": html_lib.unescape(sanction).strip(),
                "name": _text(name),
                "href": href,
                "start_date": _parse_month_date(start_text),
                "end_date": _parse_month_date(end_text or start_text),
            }
        )
    pages = [int(n) for n in re.findall(r'data-page="(\d+)"', page_html)]
    return found, (max(pages) if pages else 1)


def parse_events_page(page_html: str, sanction: str, tournament_name: str = "") -> list[dict[str, Any]]:
    """Group an events-table page into one sanction record per event code."""
    grouped: dict[str, dict[str, Any]] = {}
    for row in re.findall(r'<tr\b[^>]*role="row"[^>]*>(.*?)</tr>', page_html, flags=re.S):
        if "<th" in row:
            continue
        cells = [_text(cell) for cell in re.findall(r"<td\b[^>]*>(.*?)</td>", row, flags=re.S)]
        if len(cells) < 4:
            continue
        links = re.findall(r'href="(/event/([^"/]+)/([^"/]+)/(\d+)/[^"]*)"', row)
        if not links:
            continue
        _href, link_sanction, event_code, session_number = links[0]
        if link_sanction != sanction:
            continue
        event_id = f"{sanction}-{event_code}"
        event = grouped.get(event_id)
        if event is None:
            when = _parse_month_date(cells[0])
            event = {
                "id": event_id,
                "sanction": sanction,
                "event_code": event_code,
                "name": cells[2],
                "start_date": when.strftime("%Y-%m-%d") if when else "",
                "start_time": _clock(cells[3]),
                "session_count": 0,
                "results_available": True,
                "game_type": "Teams" if re.search(r"team|swiss", cells[2], re.I) else "Pairs",
            }
            if tournament_name:
                event["name"] = event["name"] or tournament_name
            grouped[event_id] = event
        event["session_count"] = max(int(event["session_count"]), int(session_number))
    return list(grouped.values())


def parse_handrecords_page(page_html: str, box_number: str = "") -> list[dict[str, Any]]:
    parts = re.split(r'id="board-(\d+)"', page_html)
    hands: list[dict[str, Any]] = []
    for index in range(1, len(parts), 2):
        board_number = int(parts[index])
        body = parts[index + 1]
        dealer_match = re.search(r"Dlr:\s*([A-Za-z]+)", body)
        vul_match = re.search(r"Vul:\s*([^<]+)", body)
        dealer_raw = (dealer_match.group(1) if dealer_match else "").strip()
        vul_raw = _text(vul_match.group(1) if vul_match else "")
        hand_divs = re.findall(r'<div class="hand">(.*?)</div>', body, flags=re.S)
        west = re.search(r'class="inner-slice left">(.*?)</div>', body, flags=re.S)
        east = re.search(r'class="inner-slice right">(.*?)</div>', body, flags=re.S)
        north = _suits_from_block(hand_divs[0]) if hand_divs else dict(_EMPTY_SUITS)
        south = _suits_from_block(hand_divs[-1]) if len(hand_divs) > 1 else dict(_EMPTY_SUITS)
        dd_block = ""
        dd_match = re.search(
            r'class="double-dummy">(.*?)<div class="par-score">',
            body,
            flags=re.S,
        )
        if dd_match:
            dd_block = dd_match.group(1)
        ns_match = re.search(r"NS:(.*?)EW:", dd_block, flags=re.S)
        ew_match = re.search(r"EW:(.*)", dd_block, flags=re.S)
        par_match = re.search(r'class="par-score">(.*?)</div>\s*<div', body, flags=re.S)
        hands.append(
            {
                "box_number": box_number,
                "board_number": board_number,
                "north_spades": north["spades"],
                "north_hearts": north["hearts"],
                "north_diamonds": north["diamonds"],
                "north_clubs": north["clubs"],
                "east_spades": (_suits_from_block(east.group(1)) if east else dict(_EMPTY_SUITS))["spades"],
                "east_hearts": (_suits_from_block(east.group(1)) if east else dict(_EMPTY_SUITS))["hearts"],
                "east_diamonds": (_suits_from_block(east.group(1)) if east else dict(_EMPTY_SUITS))["diamonds"],
                "east_clubs": (_suits_from_block(east.group(1)) if east else dict(_EMPTY_SUITS))["clubs"],
                "south_spades": south["spades"],
                "south_hearts": south["hearts"],
                "south_diamonds": south["diamonds"],
                "south_clubs": south["clubs"],
                "west_spades": (_suits_from_block(west.group(1)) if west else dict(_EMPTY_SUITS))["spades"],
                "west_hearts": (_suits_from_block(west.group(1)) if west else dict(_EMPTY_SUITS))["hearts"],
                "west_diamonds": (_suits_from_block(west.group(1)) if west else dict(_EMPTY_SUITS))["diamonds"],
                "west_clubs": (_suits_from_block(west.group(1)) if west else dict(_EMPTY_SUITS))["clubs"],
                "double_dummy_north_south": _dd_side(ns_match.group(1)) if ns_match else "",
                "double_dummy_east_west": _dd_side(ew_match.group(1)) if ew_match else "",
                "double_dummy_par_score": _par_from_fragment(par_match.group(1)) if par_match else "",
                "dealer": _DEALER.get(dealer_raw.upper(), dealer_raw),
                "vulnerability": _VUL.get(vul_raw.upper(), vul_raw),
            }
        )
    return hands


def _contract_from_cell(cell: str) -> str:
    suited = re.search(
        r"(\d)\s*<span class=\"([a-z]+) symbol contract\"></span>\s*(xx|x)?",
        cell,
        flags=re.I,
    )
    if suited:
        letter = _SUIT_LETTER.get(suited.group(2), "")
        return f"{suited.group(1)}{letter}{(suited.group(3) or '').lower()}"
    plain = re.sub(r"\s+", "", _text(cell)).upper()
    if plain in ("", "PASS", "P"):
        return "" if plain != "PASS" else ""
    match = re.fullmatch(r"(\d)(NT|S|H|D|C)(XX|X)?", plain)
    if not match:
        return plain
    doubled = (match.group(3) or "").lower()
    return f"{match.group(1)}{match.group(2)}{doubled}"


def _pair_from_heading(page_html: str) -> dict[str, Any]:
    match = re.search(r"<h4\b([^>]*)>(.*?)</h4>", page_html, flags=re.S)
    if not match:
        return {}
    title = ""
    title_match = re.search(r'title="([^"]*)"', match.group(1))
    if title_match:
        title = html_lib.unescape(title_match.group(1))
    heading = _text(match.group(2))
    parsed = re.match(r"(\d+)\s*(NS|EW|N-S|E-W)?\s*-\s*(.*)", heading, flags=re.I)
    names = parsed.group(3) if parsed else heading
    return {
        "pair_number": int(parsed.group(1)) if parsed else None,
        "orientation": "E-W" if parsed and (parsed.group(2) or "").upper().startswith("E") else "N-S",
        "pair_names": [part.strip() for part in names.split("&") if part.strip()],
        "pair_acbl": _acbl_numbers(title),
    }


def _selected_score_url(page_html: str) -> str:
    match = re.search(r'data-url="([^"]+/scores/[^"]+)"[^>]*selected', page_html)
    return match.group(1) if match else ""


def parse_scorecard(page_html: str, page_url: str = "") -> dict[str, Any]:
    """Parse one pair scorecard into board rows for that pair's orientation."""
    pair = _pair_from_heading(page_html)
    url = page_url or _selected_score_url(page_html)
    section = ""
    url_match = re.search(r"/scores/([^/]+)/([NE])/", url, flags=re.I)
    if url_match:
        section = url_match.group(1)
        pair["orientation"] = "N-S" if url_match.group(2).upper() == "N" else "E-W"
    rows: list[dict[str, Any]] = []
    for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", page_html, flags=re.S):
        if "board_num=" not in row_html or 'class="vs"' not in row_html:
            continue
        cells = re.findall(r"<td\b([^>]*)>(.*?)</td>", row_html, flags=re.S)
        if len(cells) < 8:
            continue
        board_match = re.search(r"board_num=(\d+)", row_html)
        if not board_match:
            continue
        if len(cells) >= 9:
            _play, _board, contract_cell, by, plus, minus, mp, pct, vs = cells[:9]
        else:
            _play, _board, contract_cell, by, plus, minus, mp, vs = cells[:8]
            pct = ("", "")
        vs_attrs, vs_html = vs
        if "vs" not in vs_attrs:
            continue
        contract = _contract_from_cell(contract_cell[1])
        plus_n = _number(_text(plus[1]))
        minus_n = _number(_text(minus[1]))
        if plus_n is not None:
            score: int | float | str | None = plus_n
        elif minus_n is not None:
            score = minus_n
        elif not contract:
            score = "PASS"
        else:
            score = None
        opponent = _text(vs_html)
        opp_match = re.match(r"(\d+)\s*-\s*(.*)", opponent)
        opp_names = opp_match.group(2) if opp_match else opponent
        bbo = _bbo_params(re.search(r'href="(https://www\.bridgebase\.com[^"]*)"', row_html).group(1)) if "bridgebase.com" in row_html else {}
        rows.append(
            {
                "board_number": int(board_match.group(1)),
                "orientation": pair.get("orientation") or "N-S",
                "contract": contract,
                "declarer": _text(by[1]).upper()[:1],
                "score": score,
                "match_points": _number(_text(mp[1])),
                "percentage": _number(_text(pct[1])),
                "pair_number": pair.get("pair_number"),
                "pair_acbl": pair.get("pair_acbl") or [],
                "pair_names": pair.get("pair_names") or [],
                "opponent_pair_number": int(opp_match.group(1)) if opp_match else None,
                "opponent_pair_names": [part.strip() for part in opp_names.split(" - ") if part.strip()],
                "_bbo": bbo,
            }
        )
    return {"section": section or "A", "rows": rows, "pair": pair}


def collect_live_links(page_html: str) -> tuple[str | None, list[str]]:
    hands = re.findall(r'href="([^"]*?/handrecords/[^"]+)"', page_html)
    scores = []
    seen: set[str] = set()
    for href in re.findall(r'href="([^"]*?/scores/[^"]+)"', page_html):
        if href in seen:
            continue
        seen.add(href)
        scores.append(href)
    return (hands[0] if hands else None), scores


def parse_session_heading(page_html: str) -> tuple[str, str, str]:
    h1 = _text(next(iter(re.findall(r"<h1\b[^>]*>(.*?)</h1>", page_html, flags=re.S)), ""))
    h2 = _text(next(iter(re.findall(r"<h2\b[^>]*>(.*?)</h2>", page_html, flags=re.S)), ""))
    when = _parse_month_date(h1.split(" - ")[0]) if h1 else None
    start_date = when.strftime("%Y%m%d") if when else ""
    description = re.sub(r"\s+Scores$", "", h2, flags=re.I).strip()
    return start_date, _clock(h1), description


def _apply_bbo_par(hands: dict[int, dict[str, Any]], row: dict[str, Any]) -> None:
    par = _par_from_bbo(row.get("_bbo") or {})
    hand = hands.get(int(row["board_number"]))
    if hand is not None and par:
        hand["double_dummy_par_score"] = par


def _hand_from_bbo(board_number: int, params: dict[str, str], box_number: str) -> dict[str, Any]:
    directions = {
        "north": _bbo_suits(params.get("n") or ""),
        "east": _bbo_suits(params.get("e") or ""),
        "south": _bbo_suits(params.get("s") or ""),
        "west": _bbo_suits(params.get("w") or ""),
    }
    record: dict[str, Any] = {
        "box_number": box_number,
        "board_number": board_number,
        "double_dummy_north_south": "",
        "double_dummy_east_west": "",
        "double_dummy_par_score": _par_from_bbo(params),
        "dealer": _DEALER.get((params.get("d") or "").upper(), ""),
        "vulnerability": _VUL.get((params.get("v") or "-").upper(), "None"),
    }
    for direction, suits in directions.items():
        for suit, cards in suits.items():
            record[f"{direction}_{suit}"] = cards
    return record


def build_session_payload(
    session_id: str,
    recap_html: str,
    hands_html: str = "",
    scorecards: list[tuple[str, str]] | None = None,
    tournament_name: str = "",
) -> dict[str, Any] | None:
    """Assemble an API-shaped session document. None when the recap has no scorecards."""
    sanction, event_code, session_number = split_session_id(session_id)
    hand_url, score_urls = collect_live_links(recap_html)
    if not score_urls and not scorecards:
        return None
    box_number = ""
    if hand_url:
        box_number = hand_url.rstrip("/").rsplit("/", 1)[-1]
    hands = {
        int(hand["board_number"]): hand
        for hand in parse_handrecords_page(hands_html, box_number=box_number)
    }
    start_date, start_time, description = parse_session_heading(recap_html)
    sections: dict[str, list[dict[str, Any]]] = {}
    for page_url, score_html in scorecards or []:
        parsed = parse_scorecard(score_html, page_url=page_url)
        if not start_date:
            start_date, start_time, description = parse_session_heading(score_html)
        bucket = sections.setdefault(parsed["section"], [])
        for row in parsed["rows"]:
            bbo = row.pop("_bbo", {}) or {}
            board_number = int(row["board_number"])
            if board_number not in hands and any(bbo.get(key) for key in ("n", "e", "s", "w")):
                hands[board_number] = _hand_from_bbo(board_number, bbo, box_number)
            _apply_bbo_par(hands, {**row, "_bbo": bbo})
            bucket.append(row)
    if not sections:
        return None
    event_id = f"{sanction}-{event_code}"
    iso_date = datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d") if start_date else ""
    tournament: dict[str, Any] = {"sanction": sanction}
    event: dict[str, Any] = {
        "id": event_id,
        "sanction": sanction,
        "event_code": event_code,
        "results_available": True,
    }
    if tournament_name:
        tournament["name"] = tournament_name
    if iso_date:
        tournament["start_date"] = iso_date
        event["start_date"] = iso_date
    if description:
        event["name"] = description
    if start_time:
        event["start_time"] = start_time
    return {
        "id": session_id,
        "session_number": session_number,
        "start_date": start_date,
        "start_time": start_time,
        "description": description,
        "box_number": box_number,
        "results_available": True,
        "tournament": tournament,
        "event": event,
        "handrecord": [hands[key] for key in sorted(hands)],
        "sections": [
            {
                "session_id": session_id,
                "section_label": label,
                "board_results": sorted(
                    rows,
                    key=lambda row: (row["board_number"], row.get("orientation") or "", row.get("pair_number") or 0),
                ),
            }
            for label, rows in sorted(sections.items())
        ],
    }


def _looks_unavailable(page_html: str) -> bool:
    lowered = (page_html or "").lower()
    return (
        "a 404 error occurred" in lowered
        or "page not found" in lowered
        or "404 not found" in lowered
    )


def _looks_gateway_error(page_html: str) -> bool:
    lowered = (page_html or "").lower()
    return "bad gateway" in lowered or "gokuprops" in lowered or "just a moment" in lowered


def _default_fetch(url: str) -> str:
    last = ""
    for attempt in range(1, 4):
        last = fetch_acbl_page(_abs_live(url))
        if _looks_gateway_error(last) and attempt < 3:
            print(f"  gateway/challenge page for {url}; retry {attempt}")
            time.sleep(2)
            continue
        return last
    return last


def _bounded(start_date: str | None, end_date: str | None) -> tuple[datetime | None, datetime | None]:
    start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
    return start, end


def download_events_from_web(
    output_dir: pathlib.Path,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
    sleep_seconds: float = 0.0,
    fetch: Callable[[str], str] | None = None,
) -> int:
    """Write ``tournaments/events/{event_id}.sanction.json`` from live.acbl.org."""
    fetcher = fetch or _default_fetch
    events_dir = output_dir.joinpath("tournaments/events")
    events_dir.mkdir(parents=True, exist_ok=True)
    start_dt, end_dt = _bounded(start_date, end_date)
    written = 0
    skipped = 0
    seen = 0
    started = time.time()
    try:
        for kind in TOURNAMENT_LIST_TYPES:
            page = 1
            max_page = 1
            while page <= max_page:
                payload = {"type": kind}
                if start_date:
                    payload["start"] = start_date
                if end_date:
                    payload["end"] = end_date
                encoded = urllib.parse.quote(json.dumps(payload, separators=(",", ":")), safe="")
                url = f"{LIVE_ORIGIN}/ajax/tourn-list/{encoded}?perPage=100&page={page}"
                print(f"[web] {kind} page {page}/{max_page}: {url}")
                page_html = fetcher(url)
                tournaments, reported_max = parse_tournament_list(page_html)
                max_page = max(max_page, reported_max)
                print(
                    f"[web] {kind} page {page}/{max_page}: "
                    f"{len(tournaments)} tournaments"
                )
                if not tournaments:
                    break
                for tournament in tournaments:
                    if not _overlaps(
                        tournament.get("start_date"),
                        tournament.get("end_date"),
                        start_dt,
                        end_dt,
                    ):
                        continue
                    if limit is not None and written >= limit:
                        return written
                    sanction = tournament["sanction"]
                    seen += 1
                    rate = (time.time() - started) / seen
                    print(
                        f"  [{seen}] rate:{rate:.1f}s/tourn "
                        f"written:{written} skipped:{skipped} "
                        f"{sanction} {tournament.get('name')}"
                    )
                    t_tourn = time.time()
                    events_html = fetcher(tournament["href"])
                    if sleep_seconds:
                        time.sleep(sleep_seconds)
                    new_here = 0
                    saved_here = 0
                    for event in parse_events_page(events_html, sanction, tournament.get("name") or ""):
                        if limit is not None and written >= limit:
                            return written
                        path = events_dir / f"{event['id']}.sanction.json"
                        sql_path = events_dir / f"{event['id']}.sanction.sql"
                        if path.exists() or sql_path.exists():
                            skipped += 1
                            saved_here += 1
                            continue
                        path.write_text(json.dumps(event, indent=2, ensure_ascii=False), encoding="utf-8")
                        written += 1
                        new_here += 1
                        print(f"  [{written}] Writing: {path.name} sessions:{event['session_count']}")
                    print(
                        f"  [{seen}] {sanction} done in {time.time() - t_tourn:.1f}s "
                        f"new:{new_here} already_saved:{saved_here} "
                        f"written:{written} skipped:{skipped}"
                    )
                page += 1
                if sleep_seconds:
                    time.sleep(sleep_seconds)
    finally:
        if fetch is None:
            shutdown_browser()
    return written


def download_sessions_from_web(
    session_ids: list[str],
    output_dir: pathlib.Path,
    sleep_seconds: float = 0.0,
    skip_if_json_exists: bool = True,
    skip_if_sql_exists: bool = True,
    fetch: Callable[[str], str] | None = None,
) -> DownloadStats:
    """Write API-shaped ``*.session.json`` files from live.acbl.org scorecards."""
    fetcher = fetch or _default_fetch
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = DownloadStats()
    total = len(session_ids)
    consecutive_fetch_failures = 0

    def _fetch_page(url: str) -> str | None:
        nonlocal consecutive_fetch_failures
        try:
            html = fetcher(url)
        except Forbidden403Error:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            consecutive_fetch_failures += 1
            print(f"ERROR fetching {LIVE_ORIGIN}{url if url.startswith('/') else '/' + url}: {exc}")
            return None
        consecutive_fetch_failures = 0
        return html

    def _give_up_on_session(session_id: str) -> bool:
        """Record a retryable miss. Stop the run after five sessions in a row fail to load."""
        stats.unavailable += 1
        stats.unavailable_ids.append(session_id)
        print(f"UNAVAILABLE (page load failed, will retry next run): {session_id}")
        if consecutive_fetch_failures >= 5:
            print("Consecutive page-load failures exceeded 5; stopping")
            stats.aborted = True
            return True
        return False

    try:
        for index, session_id in enumerate(session_ids):
            file_json = output_dir / f"{session_id}.session.json"
            file_sql = output_dir / f"{session_id}.session.sql"
            if skip_if_sql_exists and file_sql.exists():
                print(f"{index}/{total}: File exists: {file_sql}: skipping")
                stats.skipped += 1
                continue
            if skip_if_json_exists and file_json.exists():
                print(f"{index}/{total}: File exists: {file_json}: skipping")
                stats.skipped += 1
                continue
            try:
                sanction, event_code, session_number = split_session_id(session_id)
            except ValueError as exc:
                print(f"ERROR: {exc}")
                stats.errors += 1
                continue
            recap_url = f"/event/{sanction}/{event_code}/{session_number}/recap"
            print(f"{index}/{total} [web] {LIVE_ORIGIN}{recap_url}")
            try:
                recap_html = _fetch_page(recap_url)
            except Forbidden403Error as exc:
                print(f"ERROR: {exc}")
                stats.errors += 1
                stats.aborted = True
                break
            if recap_html is None:
                if _give_up_on_session(session_id):
                    break
                continue
            if _looks_unavailable(recap_html):
                stats.unavailable += 1
                stats.unavailable_ids.append(session_id)
                print(f"UNAVAILABLE (webpage 404): {session_id}")
                continue
            hand_url, score_urls = collect_live_links(recap_html)
            if not score_urls:
                stats.unavailable += 1
                stats.unavailable_ids.append(session_id)
                print(f"UNAVAILABLE (no pair scorecards): {session_id}")
                continue
            hands_html = ""
            if hand_url:
                try:
                    loaded = _fetch_page(hand_url)
                    hands_html = loaded or ""
                except Forbidden403Error as exc:
                    print(f"  handrecords skipped: {exc}")
            scorecards: list[tuple[str, str]] = []
            failed = False
            score_failed = False
            for score_url in score_urls:
                try:
                    loaded = _fetch_page(score_url)
                except Forbidden403Error as exc:
                    print(f"ERROR: {exc}")
                    stats.errors += 1
                    stats.aborted = True
                    failed = True
                    break
                if loaded is None:
                    score_failed = True
                    break
                scorecards.append((score_url, loaded))
                if sleep_seconds:
                    time.sleep(sleep_seconds)
            if failed:
                break
            if score_failed:
                if _give_up_on_session(session_id):
                    break
                continue
            payload = build_session_payload(session_id, recap_html, hands_html, scorecards)
            if not payload or not any(section["board_results"] for section in payload["sections"]):
                stats.unavailable += 1
                stats.unavailable_ids.append(session_id)
                print(f"UNAVAILABLE (no board rows): {session_id}")
                continue
            text = json.dumps(payload, indent=4, ensure_ascii=False)
            file_json.write_text(text, encoding="utf-8")
            stats.written += 1
            print(
                f"{index}/{total}: Writing:{file_json.name} "
                f"boards:{sum(len(section['board_results']) for section in payload['sections'])} "
                f"hands:{len(payload['handrecord'])}"
            )
    finally:
        if fetch is None:
            shutdown_browser()
    return stats
