
"""
Created on Mon Nov 03 08:54:55 2025

@author: andresardao
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flask HTTP layer for the recommender service.

Key responsibilities:
- Provide stable JSON responses (NumPy/Pandas/datetime-safe).
- Expose recommendation endpoint.
- Expose operational endpoints (health, cache status/rebuild).
- Integrate with the semantic agent by updating TABS on demand.
"""

import json
import traceback
import datetime
from typing import Any, Dict

from flask import Flask, request, current_app, Response
from flask.json.provider import DefaultJSONProvider

# --- Domain/service imports (keep these minimal and explicit) ---
# Expectation:
# - services_recommender.py exports: recommender(data_challenge_id: int) -> dict, and global TABS: Dict[str, list[str]]
# - app_infra_redis.py exports: redis_cli (redis client), cache_rebuild(yr:int, sub:int)->dict, _cache_key_for(yr,sub)->str
from services_recommender import recommender, TABS
from app_infra_redis import redis_cli, cache_rebuild, _cache_key_for

from werkzeug.exceptions import HTTPException


####################################
# JSON helpers (NumPy/Pandas-safe) #
####################################

def _json_default(o: Any):
    """
    Fallback serializer for types not supported by the standard json encoder.
    Handles common NumPy/Pandas/datetime objects.
    """
    # NumPy scalars / arrays
    try:
        import numpy as np  # local import to avoid hard dependency if not used
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass

    # Pandas / datetime types
    try:
        import pandas as pd  # local import to avoid hard dependency if not used
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
    except Exception:
        pass

    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()

    # Last resort
    return str(o)



def _json_sanitize(x: Any) -> Any:
    # numpy scalars -> python scalars
    try:
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass

    # pandas timestamps, etc.
    try:
        import pandas as pd
        if isinstance(x, pd.Timestamp):
            return x.isoformat()
    except Exception:
        pass

    if isinstance(x, dict):
        return {str(_json_sanitize(k)) if not isinstance(k, (str, int, float, bool)) and k is not None else _json_sanitize(k): _json_sanitize(v)
                for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_json_sanitize(i) for i in x]

    return x

def json_response(payload: dict, status: int = 200) -> Response:
    payload = _json_sanitize(payload)
    return Response(
        json.dumps(payload, ensure_ascii=False),
        status=status,
        mimetype="application/json; charset=utf-8",
    )
class NumpyJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        # Ensures our JSON converter is used everywhere Flask serializes JSON.
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("default", _json_default)
        return super().dumps(obj, **kwargs)

    def loads(self, s, **kwargs):
        return super().loads(s, **kwargs)


####################
# App factory ######
####################

def create_app() -> Flask:
    """
    Create a Flask app instance (WSGI-friendly).
    """
    app = Flask(__name__)
    app.json = NumpyJSONProvider(app)

    # -------- Recommendations --------

    @app.post("/get_recommendations")
    def get_recommendations():
        """
        Contract:
        - Input: {"data_challenge_id": <int>}
        - Output: recommender payload (dict)
        """
        data = request.get_json(silent=True) or request.form or {}
        try:
            data_challenge_id = int(data.get("data_challenge_id"))
        except Exception:
            return json_response({"error": "field 'data_challenge_id' (int) is required"}, 400)

        try:
            recs = recommender(data_challenge_id)
            return json_response(recs, 200)
        except Exception as e:
            current_app.logger.exception("/get_recommendations failed")
            return json_response(
                {"error": str(e), "traceback": traceback.format_exc().splitlines()[-12:]},
                500
            )

    # -------- Semantic agent integration (TABS update) --------

    @app.post("/tabs/update")
    def tabs_update():
        """
        Contract (called by the semantic agent service when new error tables are generated):
        Input:
          {
            "hab": "EF0XMAYZ",
            "tables": [
              "classificacao_erros_EF0XMAYZ.csv",
              "classificacao_erros_<COMBO_PRE>.csv"
            ]
          }

        Side effect:
          TABS["EF0XMAYZ"] = ["classificacao_erros_EF0XMAYZ.csv", "classificacao_erros_<COMBO_PRE>.csv"]
        """
        data = request.get_json(silent=True) or {}
        hab = data.get("hab")
        tables = data.get("tables")

        if not isinstance(hab, str) or not hab.strip():
            return json_response({"error": "field 'hab' (non-empty string) is required"}, 400)
        if not isinstance(tables, list) or not all(isinstance(x, str) and x.strip() for x in tables):
            return json_response({"error": "field 'tables' must be a list of non-empty strings"}, 400)

        # Enforce the exact format requested.
        TABS[hab] = tables

        return json_response({"status": "ok", "hab": hab, "tables": TABS[hab]}, 200)

    # -------- Diagnostics --------

    @app.get("/__json_test")
    def _json_test():
        """
        Quick JSON serialization test (NumPy/Pandas/datetime).
        """
        try:
            import numpy as np
            return json_response(
                {
                    "np_int": np.int64(7),
                    "np_float": np.float64(3.14),
                    "np_arr": np.array([1, 2, 3]),
                    "dt": datetime.datetime.utcnow(),
                },
                200,
            )
        except Exception as e:
            return json_response({"error": str(e)}, 500)

    # -------- /cache/status (robust; does not deserialize the cached bundle) --------

    @app.get("/cache/status")
    def cache_status():
        """
        Query params:
          - yr: int
          - sub: int

        Returns:
          - redis key, exists flag, ttl, strlen
        """
        try:
            yr = request.args.get("yr", type=int)
            sub = request.args.get("sub", type=int)
            if yr is None or sub is None:
                return json_response({"error": "params 'yr' and 'sub' (ints) are required"}, 400)

            key = _cache_key_for(yr, sub)
            exists = bool(redis_cli.exists(key))
            if not exists:
                return json_response({"yr": yr, "sub": sub, "key": key, "exists": False}, 200)

            ttl = redis_cli.ttl(key)
            size = redis_cli.strlen(key)

            return json_response(
                {
                    "yr": yr,
                    "sub": sub,
                    "key": key,
                    "exists": True,
                    "ttl_seconds": ttl,
                    "value_bytes": size,
                },
                200,
            )
        except Exception as e:
            current_app.logger.exception("/cache/status failed")
            return json_response(
                {"error": str(e), "traceback": traceback.format_exc().splitlines()[-12:]},
                500,
            )

    # -------- /cache/rebuild --------

    @app.post("/cache/rebuild")
    def cache_rebuild_http():
        """
        Input (JSON):
          {"yr": <int>, "sub": <int>}

        Rebuilds the cached features bundle. Response includes only a summary (keys),
        not the full bundle.
        """
        try:
            data = request.get_json(silent=True) or {}
            yr = int(data["yr"])
            sub = int(data["sub"])
        except Exception:
            return json_response({"error": "fields 'yr' and 'sub' (ints) are required"}, 400)

        try:
            bundle = cache_rebuild(yr, sub)  # returns a large dict
            return json_response({"status": "ok", "yr": yr, "sub": sub, "keys": list(bundle.keys())}, 200)
        except Exception as e:
            current_app.logger.exception("/cache/rebuild failed")
            return json_response(
                {"error": str(e), "traceback": traceback.format_exc().splitlines()[-12:]},
                500,
            )

    # -------- /health --------

    @app.get("/health")
    def health():
        """
        Minimal healthcheck (probes Redis).
        By default returns 200 even if degraded; change to 500 if you want strict behavior.
        """
        try:
            redis_ok = True
            try:
                redis_cli.ping()
            except Exception:
                redis_ok = False

            payload = {"status": "ok" if redis_ok else "degraded", "redis_ok": redis_ok}
            return json_response(payload, 200)
        except Exception as e:
            current_app.logger.exception("/health failed")
            return json_response(
                {"error": str(e), "traceback": traceback.format_exc().splitlines()[-12:]},
                500,
            )
        
    @app.get("/")
    def root():
        return health()


    # -------- Global error handler (JSON) --------

    @app.errorhandler(Exception)
    def handle_exception(e):
        # Erros HTTP legítimos (404, 405, 400, etc.)
        if isinstance(e, HTTPException):
            return json_response(
                {"error": e.description},
                e.code,
            )
    
        # Erros reais (bugs)
        current_app.logger.exception("unhandled exception")
        return json_response(
            {
                "error": str(e),
                "traceback": traceback.format_exc().splitlines()[-12:],
            },
            500,
        )
    
    return app

# WSGI entrypoint
app = create_app()

if __name__ == "__main__":
    # Run the server (default: port 8000).
    app.run(host="0.0.0.0", port=8000)

