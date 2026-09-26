"""FastAPI service exposing unified ACBL club and tournament results.

Scraping, cache, and table-building live here. The MCP server is a thin
HTTP client of this API (see ../MortyBridgeMCP).

  python acbl_club_api_server.py
  GET http://127.0.0.1:8508/docs
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, Query, Response
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import acbl_club_api_service as svc

ACBL_CLUB_API_PORT = int(os.environ.get("ACBL_CLUB_API_PORT", "8508"))

app = FastAPI(
    title="ACBL Results API",
    description=(
        "Shared club and tournament listings plus augmented postmortems. "
        "Historical parquet, API cache, and live sources use one interface."
    ),
    version="1.0.0",
)
# Wildcard origins cannot be combined with credentials per the CORS spec,
# and nothing here uses cookies, so credentials stay disabled.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(svc.ClubApiError)
async def club_api_error_handler(request, exc: svc.ClubApiError) -> JSONResponse:
    body = {"detail": exc.detail}
    if exc.hint:
        body["hint"] = exc.hint
    return JSONResponse(status_code=exc.status_code, content=body)


@app.get("/health")
def health() -> dict:
    info = svc.dataset_info()
    return {"status": "ok", "service": "acbl-results-api", **info}


@app.get("/tournaments/info")
def get_tournament_dataset_info() -> dict:
    return {
        "status": "ok",
        "service": "acbl-results-api",
        **svc.tournament_dataset_info(),
    }


@app.get("/clubs/info")
def get_club_dataset_info(response: Response) -> dict:
    response.headers["Cache-Control"] = "no-store"
    return {
        "status": "ok",
        "service": "acbl-results-api",
        **svc.club_dataset_info(),
    }


@app.get("/clubs")
def get_clubs(
    q: Optional[str] = Query(
        None, description="Typo-tolerant filter on club id/name/location"
    ),
    limit: int = Query(svc.DEFAULT_ROW_LIMIT, ge=1, le=svc.MAX_ROW_LIMIT),
    refresh: bool = Query(False),
) -> dict:
    return svc.club_list(query=q, limit=limit, refresh=refresh)


@app.get("/clubs/{club_id}")
def get_club(club_id: str, refresh: bool = Query(False)) -> dict:
    return svc.club_info(club_id, refresh=refresh)


@app.get("/clubs/{club_id}/games")
def get_club_games(
    club_id: str,
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD inclusive"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD inclusive"),
    limit: int = Query(svc.DEFAULT_ROW_LIMIT, ge=1, le=svc.MAX_ROW_LIMIT),
    refresh: bool = Query(False),
) -> dict:
    return svc.club_games(
        club_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        refresh=refresh,
    )


@app.get("/players/lookup")
def get_player_lookup(
    q: str = Query(..., description="Player name substring or digits-only ACBL number"),
    session_id: Optional[str] = Query(None),
    club_id: Optional[str] = Query(None),
    limit: int = Query(svc.DEFAULT_ROW_LIMIT, ge=1, le=svc.MAX_ROW_LIMIT),
) -> dict:
    return svc.player_lookup(query=q, session_id=session_id, club_id=club_id, limit=limit)


@app.get("/players/{player_id}/games")
def get_player_games(
    player_id: str,
    limit: int = Query(svc.DEFAULT_ROW_LIMIT, ge=1, le=svc.MAX_ROW_LIMIT),
    refresh: bool = Query(False),
) -> dict:
    return svc.player_games(player_id, limit=limit, refresh=refresh)


@app.get("/clubs/players/{player_id}/postmortems")
def get_historical_club_player_sessions(
    player_id: str,
    limit: int = Query(svc.DEFAULT_ROW_LIMIT, ge=1, le=svc.MAX_ROW_LIMIT),
) -> dict:
    return svc.historical_club_player_sessions(player_id, limit=limit)


@app.get("/tournaments/players/{player_id}/sessions")
def get_tournament_player_sessions(
    player_id: str,
    limit: int = Query(svc.DEFAULT_ROW_LIMIT, ge=1, le=svc.MAX_ROW_LIMIT),
    refresh: bool = Query(False),
    tournament_name: Optional[str] = Query(
        None, description="Typo-tolerant tournament/event name filter"
    ),
) -> dict:
    return svc.tournament_player_sessions(
        player_id,
        limit=limit,
        refresh=refresh,
        tournament_name=tournament_name,
    )


@app.get("/sessions/{session_id}/tables")
def get_session_tables(
    session_id: str,
    refresh: bool = Query(False),
    allow_live: bool = Query(True),
) -> dict:
    return svc.session_tables(
        session_id, refresh=refresh, allow_live=allow_live)


@app.get("/sessions/{session_id}/tables/{table_name}")
def get_session_table(
    session_id: str,
    table_name: str,
    columns: Optional[str] = Query(None, description="Comma-separated column names"),
    limit: int = Query(svc.DEFAULT_ROW_LIMIT, ge=1, le=svc.MAX_ROW_LIMIT),
    refresh: bool = Query(False),
    allow_live: bool = Query(True),
) -> dict:
    return svc.session_results(
        session_id,
        table=table_name,
        columns=columns,
        limit=limit,
        refresh=refresh,
        allow_live=allow_live,
    )


@app.get("/sessions/{session_id}/raw")
def get_session_raw(
    session_id: str, response: Response, refresh: bool = Query(False)
) -> dict:
    """Verbatim details JSON (nested), for the postmortem augmentation pipeline."""
    data, source = svc.session_raw_with_source(session_id, refresh=refresh)
    response.headers["X-ACBL-Data-Source"] = source
    return data


@app.get("/sessions/{session_id}/frames")
def get_session_frames(session_id: str, refresh: bool = Query(False)) -> dict:
    """Flat session tables for postmortem processing, including parquet fallback."""
    return svc.session_frames_payload(session_id, refresh=refresh)


@app.get("/sessions/{session_id}/postmortem.parquet")
def get_session_postmortem_parquet(
    session_id: str,
    player_id: Optional[str] = Query(None),
    refresh: bool = Query(False),
) -> Response:
    """Compatibility route for a complete API-resolved postmortem."""
    payload, meta = svc.session_augmented_parquet(
        session_id, player_id=player_id, refresh=refresh)
    return Response(
        content=payload,
        media_type="application/vnd.apache.parquet",
        headers={
            "Content-Disposition": (
                f'inline; filename="acbl-postmortem-{session_id}.parquet"'
            ),
            "X-ACBL-Data-Source": str(meta["source"]),
            "X-ACBL-Data-Updated": str(meta["fetched_at"] or ""),
            "X-ACBL-Data-Tier": str(meta.get("source_tier") or ""),
        },
    )


@app.get("/postmortems/{session_id}.parquet")
def get_postmortem_parquet(
    session_id: str,
    player_id: Optional[str] = Query(None),
    refresh: bool = Query(False),
) -> Response:
    return get_session_postmortem_parquet(
        session_id, player_id=player_id, refresh=refresh)


@app.get("/tournaments/sessions/{session_id}/postmortem.parquet")
def get_tournament_postmortem_parquet(
    session_id: str,
    player_id: Optional[str] = Query(None),
    refresh: bool = Query(False),
) -> Response:
    return get_session_postmortem_parquet(
        session_id, player_id=player_id, refresh=refresh)


@app.get("/postmortems/{session_id}/boards")
def get_postmortem_boards(
    session_id: str,
    player_id: str = Query(...),
    only_my_boards: bool = Query(True),
    columns: Optional[str] = Query(None),
    limit: int = Query(svc.DEFAULT_ROW_LIMIT, ge=1, le=svc.MAX_ROW_LIMIT),
    refresh: bool = Query(False),
) -> dict:
    return svc.postmortem_boards(
        session_id,
        player_id,
        only_my_boards=only_my_boards,
        columns=columns,
        limit=limit,
        refresh=refresh,
    )


@app.get("/postmortems/{session_id}/sql")
def get_postmortem_sql(
    session_id: str,
    player_id: str = Query(...),
    sql: str = Query(...),
    limit: int = Query(svc.DEFAULT_ROW_LIMIT, ge=1, le=svc.MAX_ROW_LIMIT),
    refresh: bool = Query(False),
) -> dict:
    return svc.postmortem_sql(
        session_id,
        player_id,
        sql,
        limit=limit,
        refresh=refresh,
    )


@app.get("/postmortems/{session_id}/schema")
def get_postmortem_schema(
    session_id: str,
    player_id: str = Query(...),
    pattern: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=svc.MAX_ROW_LIMIT),
    refresh: bool = Query(False),
) -> dict:
    return svc.postmortem_schema(
        session_id,
        player_id,
        pattern=pattern,
        limit=limit,
        refresh=refresh,
    )


@app.get("/sessions/{session_id}/sql")
def get_session_sql(
    session_id: str,
    sql: str = Query(..., description="DuckDB SQL against registered session tables"),
    limit: int = Query(svc.DEFAULT_ROW_LIMIT, ge=1, le=svc.MAX_ROW_LIMIT),
    refresh: bool = Query(False),
    allow_live: bool = Query(True),
) -> dict:
    return svc.session_sql(
        session_id,
        sql=sql,
        limit=limit,
        refresh=refresh,
        allow_live=allow_live,
    )


class BrowserHandoffBody(BaseModel):
    url: str
    html: str = Field(min_length=1, max_length=svc.HANDOFF_MAX_CHARS)


@app.post("/handoff")
def post_browser_handoff(body: BrowserHandoffBody) -> dict:
    """Cache a my.acbl.org page loaded in the caller's browser."""
    return svc.ingest_browser_handoff(body.url, body.html)


@app.get("/sessions/{session_id}")
def get_session_default(
    session_id: str,
    table: str = Query("board_results"),
    columns: Optional[str] = None,
    limit: int = Query(svc.DEFAULT_ROW_LIMIT, ge=1, le=svc.MAX_ROW_LIMIT),
    refresh: bool = Query(False),
    allow_live: bool = Query(True),
) -> dict:
    return svc.session_results(
        session_id,
        table=table,
        columns=columns,
        limit=limit,
        refresh=refresh,
        allow_live=allow_live,
    )


if __name__ == "__main__":
    import uvicorn

    print(
        f"[acbl-results-api] starting on :{ACBL_CLUB_API_PORT}; cache -> {svc.CACHE_DIR}",
        flush=True,
    )
    uvicorn.run(
        "acbl_club_api_server:app",
        host="0.0.0.0",
        port=ACBL_CLUB_API_PORT,
        reload=False,
    )
