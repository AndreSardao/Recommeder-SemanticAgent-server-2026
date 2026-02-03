"""
Created on Mon Jan 19 01:23:13 2026

@author: andresardao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib import request as urlrequest

from flask import Flask, request, Response

from error_table_agent_semantic_v6_bundles import run_agent_local, load_curated_from_bundles
from app_infra_redis import get_features  # uses same Redis infra as recommender


########################
# JSON response helper #
########################
def json_response(payload: Dict[str, Any], status: int = 200) -> Response:
    return Response(
        json.dumps(payload, ensure_ascii=False),
        status=status,
        mimetype="application/json; charset=utf-8",
    )


def _post_json(url: str, payload: Dict[str, Any], timeout_seconds: int = 15) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout_seconds) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except Exception:
            return {"raw": raw, "status": getattr(resp, "status", None)}


##################################
# BNCC -> (grade_id, subject_id) #
##################################
BNCC_SUBJECT_MAP = {
    "LP": 1,
    "MA": 2,
    "CI": 3,
    "HI": 4,
    "LI": 6,
    "GE": 7,
}

_BNCC_RE = re.compile(r"^EF(\d{2})([A-Z]{2})\d+$")


def bncc_to_grade_and_subject(hab: str) -> Tuple[int, int]:
    """
    Parses BNCC skill code like 'EF09HI16' and returns (grade_id, subject_id).
    grade_id: int from the two digits after 'EF' (e.g., '09' -> 9)
    subject_id: mapping from the two-letter component (e.g., 'HI' -> 4)
    """
    hab = (hab or "").strip().upper()
    m = _BNCC_RE.match(hab)
    if not m:
        raise ValueError(f"Invalid BNCC code format: {hab}")

    grade = int(m.group(1))
    sub_code = m.group(2)
    if sub_code not in BNCC_SUBJECT_MAP:
        raise ValueError(f"Unsupported BNCC subject code '{sub_code}' in {hab}")

    return grade, int(BNCC_SUBJECT_MAP[sub_code])


#############
# Flask app #
#############
def create_app() -> Flask:
    app = Flask(__name__)

    # Output base folder:
    # - tables (final error tables) go to OUTDIR
    # - new listings go to OUTDIR/novas
    OUTDIR = Path(os.getenv("AGENT_OUTDIR", "./error_tables_out")).resolve()

    # Curated bundles JSON used by the semantic agent
    CURATED_BUNDLES_JSON = os.getenv("CURATED_BUNDLES_JSON", "").strip()

    # Recommender integration
    RECOMMENDER_TABS_UPDATE_URL = os.getenv(
        "RECOMMENDER_TABS_UPDATE_URL",
        "http://127.0.0.1:8000/tabs/update"
    ).strip()

    # LLM settings
    DEFAULT_MODEL = os.getenv("AGENT_MODEL", "gpt-5.1")
    DEFAULT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.0"))
    DEFAULT_MIN_CONF = float(os.getenv("AGENT_MIN_CONFIDENCE", "0.35"))

    @app.get("/health")
    def health():
        return json_response(
            {
                "status": "ok",
                "outdir": str(OUTDIR),
                "curated_bundles_json_set": bool(CURATED_BUNDLES_JSON),
                "recommender_tabs_update_url": RECOMMENDER_TABS_UPDATE_URL,
            },
            200,
        )

    @app.post("/generate_error_tables")
    def generate_error_tables():
        """
        Contract (SGP -> agent app):
          Input JSON:
            {
              "hab": "EF05MA05",
              "notify_recommender": true,     # optional (default true)
              "outdir": "/custom/outdir",     # optional
              "model": "gpt-5.1",             # optional
              "temperature": 0.0,             # optional
              "min_confidence": 0.35,         # optional
              "new_listings": ["...csv"...]   # optional override (if omitted, generated automatically)
            }

          Output JSON (agent app -> SGP):
            {
              "hab": "EF05MA05",
              "tables": ["classificacao_erros_EF05MA05.csv", "classificacao_erros_<COMBO_PRE>.csv?"],
              "notify": { ... }               # diagnostics; SGP can ignore
            }
        """
            
        data = request.get_json(silent=True) or {}

        hab = (data.get("hab") or "").strip().upper()
        if not hab:
            return json_response({"error": "field 'hab' (non-empty string) is required"}, 400)

        notify = bool(data.get("notify_recommender", True))
        outdir = Path((data.get("outdir") or str(OUTDIR))).resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        model = str(data.get("model") or DEFAULT_MODEL)
        temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
        min_confidence = float(data.get("min_confidence", DEFAULT_MIN_CONF))

        # Optional override: allow passing listing files explicitly (useful for debugging)
        new_listings = data.get("new_listings") or []
        if new_listings and (not isinstance(new_listings, list) or not all(isinstance(x, str) and x.strip() for x in new_listings)):
            return json_response({"error": "field 'new_listings' must be a list of file paths (or omitted)"}, 400)

        if not CURATED_BUNDLES_JSON:
            return json_response(
                {"error": "CURATED_BUNDLES_JSON env var is required (path to curated bundles JSON)"},
                500,
            )

        try:
            # 1) BNCC -> (yr, sub)
            yr, sub = bncc_to_grade_and_subject(hab)

            # 2) Features bundle (from recommender infra/Redis)
            feats = get_features(yr, sub)

            # 3) Generate listings (if not provided)
            if not new_listings:
                from Listing_maker import maestro
                m = maestro(hab, feats=feats, out_dir=str(outdir / "novas"))
              
                # listings (files)
                new_listings = [m["hab_csv"]] + ([m["prereqs_csv"]] if m.get("prereqs_csv") else [])
                
                # prereqs codes (metadata) — Hygiene
                prereqs_codes = m.get("prereqs_codes") or []
                prereqs_codes = sorted(set(prereqs_codes))  # kills duplicates (ex.: 09,09,10)
                
                # se não há listagem de prereq, então não há prereq para o agente
                if not m.get("prereqs_csv"):
                    prereqs_codes = []

            # 4) Load curated bundles for the semantic agent
            curated_tables_files, curated_listings_files = load_curated_from_bundles(Path(CURATED_BUNDLES_JSON))

            # 5) Run semantic agent to produce the final error tables
            hab_path, pre_path = run_agent_local(
                hab=hab,
                prereqs=prereqs_codes,  # optional: if your agent can infer prereqs from listings, keep empty
                curated_tables_files=curated_tables_files,
                curated_listings_files=curated_listings_files,
                new_listings_files=[Path(p) for p in new_listings],
                outdir=outdir,
                model=model,
                temperature=temperature,
                min_confidence=min_confidence,
            )

            # 6) Prepare stable table names i.e. what will be stored in TABS
            def as_path(p: Union[str, Path]) -> Path:
                return p if isinstance(p, Path) else Path(p)
            
            hab_path = as_path(hab_path)          # <- garante Path
            tables: List[str] = [hab_path.name]
            if pre_path is not None:
                pre_path = as_path(pre_path)
                tables.append(pre_path.name)
            
            # 7) Update recommender TABS (best-effort; do not fail table generation if recommender is down)
            notify_payload = {"hab": hab, "tables": tables}
            notify_result = None
            notify_ok = False

            if notify:
                try:
                    notify_result = _post_json(
                        RECOMMENDER_TABS_UPDATE_URL,
                        notify_payload,
                        timeout_seconds=20,
                    )
                    notify_ok = True
                except Exception as e:
                    notify_result = {"error": str(e)}
                    notify_ok = False

            # 8) Response to SGP (contract)
            response_payload = {
                "hab": hab,
                "tables": tables,
                "notify": {
                    "attempted": bool(notify),
                    "ok": bool(notify_ok),
                    "recommender_url": RECOMMENDER_TABS_UPDATE_URL,
                    "result": notify_result,
                },
            }
            return json_response(response_payload, 200)

        except Exception as e:
            app.logger.exception("/generate_error_tables failed")
            return json_response(
                {"error": str(e), "traceback": traceback.format_exc().splitlines()[-25:]},
                500,
            )

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    app.run(host="0.0.0.0", port=port)
