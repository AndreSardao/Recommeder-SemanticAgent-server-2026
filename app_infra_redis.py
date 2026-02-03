"""
Created on Sun Oct 19 07:39:19 2025

@author: andresardao
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# app_infra_redis.py
from __future__ import annotations

import json
import os
import time
import zlib
import pickle
from typing import Any, Optional

import redis

# Import local: the feature builder that replaces _gera_todos_os_objetos
from services_features import _gera_todos_os_objetos


#########################################
# Redis client (single instance)  #######
#########################################

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "3.0"))
REDIS_DECODE_RESPONSES = False  # keep bytes for binary payloads

redis_cli = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    socket_timeout=REDIS_SOCKET_TIMEOUT,
    decode_responses=REDIS_DECODE_RESPONSES,
)

# Namespaces/keys
FEATURES_KEY_PREFIX = os.getenv("FEATURES_KEY_PREFIX", "recommender:features")
TABS_HASH_KEY = os.getenv("TABS_HASH_KEY", "recommender:tabs")  # hash: field=hab, value=json(list[str])

# Cache TTL (seconds)
FEATURES_TTL_SECONDS = int(os.getenv("FEATURES_TTL_SECONDS", str(60 * 60 * 24)))  # default: 24h

# Compression
COMPRESS_LEVEL = int(os.getenv("REDIS_COMPRESS_LEVEL", "6"))
COMPRESS_MIN_BYTES = int(os.getenv("REDIS_COMPRESS_MIN_BYTES", "1024"))  # only compress if worth it

# Serialization versioning (defensive)
FEATURES_PAYLOAD_VERSION = 1

# Cache: uma key por par (yr, sub)
CACHE_TTL         = int(os.environ.get('CACHE_TTL', 24 * 3600))
CACHE_KEY_PREFIX  = os.environ.get('CACHE_KEY_PREFIX', 'recommender:features')

# opcional: flags de controle
ENABLE_COMPRESSION = True
COMPRESS_LEVEL = 3  # 1..9; 3 é bom custo/benefício
MAGIC = b'Z1'       # prefixo para identificar payloads comprimidos


###################
# Utilities #######
###################

def _cache_key_for(yr: int, sub: int) -> str:
    """
    Canonical Redis key for the feature bundle of (year, subject).
    """
    return f"{FEATURES_KEY_PREFIX}:{int(yr)}:{int(sub)}"


def _normalize_bytes(x: Any) -> bytes:
    if isinstance(x, bytes):
        return x
    if isinstance(x, bytearray):
        return bytes(x)
    if isinstance(x, str):
        return x.encode("utf-8")
    return str(x).encode("utf-8")


def _json_dumps(obj):
    """
    JSON serializer for Redis cache.
    Normalizes numpy/pandas types, including dict keys (JSON forbids numpy scalar keys).
    """
    def _norm(x):
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

        try:
            import pandas as pd
            if isinstance(x, pd.Timestamp):
                return x.isoformat()
        except Exception:
            pass

        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                kk = _norm(k)
                if not isinstance(kk, (str, int, float, bool)) and kk is not None:
                    kk = str(kk)
                out[kk] = _norm(v)
            return out

        if isinstance(x, (list, tuple)):
            return [_norm(i) for i in x]

        return x

    return json.dumps(_norm(obj), ensure_ascii=False)



def _json_loads(s: bytes | str) -> Any:
    if isinstance(s, (bytes, bytearray)):
        s = s.decode("utf-8", errors="ignore")
    return json.loads(s)


def _maybe_compress(raw: bytes) -> tuple[bytes, bool]:
    """
    Returns (payload, compressed_flag).
    """
    if len(raw) < COMPRESS_MIN_BYTES:
        return raw, False
    return zlib.compress(raw, level=COMPRESS_LEVEL), True


def _maybe_decompress(payload: bytes, compressed: bool) -> bytes:
    return zlib.decompress(payload) if compressed else payload


####################################################################
# Feature bundle cache (save/load/ttl/rebuild)  ####################
####################################################################

def _save_features(yr: int, sub: int, bundle: dict, ttl_seconds: int | None = None) -> dict:
    payload = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
    if ENABLE_COMPRESSION:
        payload = MAGIC + zlib.compress(payload, COMPRESS_LEVEL)
    redis_cli.set(_cache_key_for(yr, sub), payload, ex=(ttl_seconds or CACHE_TTL))
    return {"cached": True, "yr": yr, "sub": sub}


def _load_features(yr: int, sub: int):
    """Lê do Redis e desserializa. Compatível com chaves antigas sem compressão."""
    raw = redis_cli.get(_cache_key_for(yr, sub))
    if not raw:
        return None
    # Detecta pelo prefixo; mantém fallback para casos antigos
    try:
        if raw.startswith(MAGIC):
            return pickle.loads(zlib.decompress(raw[len(MAGIC):]))
        return pickle.loads(raw)  # legacy: sem compressão
    except Exception:
        # fallback extra: caso exista alguma chave comprimida sem prefixo (muito raro)
        try:
            return pickle.loads(zlib.decompress(raw))
        except Exception as e:
            # se der ruim, propaga para aparecer no log
            raise e

def _ttl_features(yr: int, sub: int) -> int:
    """
    Redis TTL semantics:
      -2: key does not exist
      -1: key exists but has no associated expire
    """
    key = _cache_key_for(yr, sub)
    return int(redis_cli.ttl(key))


def cache_rebuild(yr: int, sub: int, ttl_seconds: int = FEATURES_TTL_SECONDS) -> dict[str, Any]:
    """
    Forces rebuild of (yr, sub) feature bundle and saves into Redis.
    """
    bundle = _gera_todos_os_objetos(int(yr), int(sub))
    meta = _save_features(int(yr), int(sub), bundle=bundle, ttl_seconds=ttl_seconds)
    return {"meta": meta, "bundle": bundle}


def get_features(yr: int, sub: int, ttl_seconds: int = FEATURES_TTL_SECONDS) -> dict[str, Any]:
    """
    Read-through cache:
      - if present in Redis: returns it
      - else rebuilds and stores
    """
    cached = _load_features(int(yr), int(sub))
    if cached is not None:
        return cached

    rebuilt = cache_rebuild(int(yr), int(sub), ttl_seconds=ttl_seconds)
    return rebuilt["bundle"]


###########################################################################
# TABS registry (keeps your mental model: TABS[hab] = [csv1, csv2, ...]) ##
###########################################################################

def tabs_set(hab: str, tables: list[str]) -> None:
    """
    Logical equivalent to: TABS[hab] = tables
    Persisted as a Redis hash entry:
      HSET recommender:tabs <hab> '["file1.csv","file2.csv"]'
    """
    if not isinstance(hab, str) or not hab.strip():
        raise ValueError("tabs_set: 'hab' must be a non-empty string")
    if not isinstance(tables, list) or not tables or not all(isinstance(x, str) and x.strip() for x in tables):
        raise ValueError("tabs_set: 'tables' must be a non-empty list[str]")

    hab = hab.strip()
    value = _json_dumps([t.strip() for t in tables])
    redis_cli.hset(TABS_HASH_KEY, hab, _normalize_bytes(value))


def tabs_get(hab: str) -> Optional[list[str]]:
    """
    Returns the tables list or None if not registered.
    """
    if not isinstance(hab, str) or not hab.strip():
        return None

    raw = redis_cli.hget(TABS_HASH_KEY, hab.strip())
    if not raw:
        return None

    try:
        arr = _json_loads(raw)
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            return arr
        return None
    except Exception:
        return None


def tabs_has(hab: str) -> bool:
    if not isinstance(hab, str) or not hab.strip():
        return False
    return bool(redis_cli.hexists(TABS_HASH_KEY, hab.strip()))


####################################################################
# Celery (optional but production-ready inside this module)  #######
####################################################################
# I am keeping Celery here bc it avoids creating a 6th module, but still supports scheduled rebuilds.

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"))
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)

try:
    from celery import Celery  # type: ignore

    cel = Celery("recommender", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

    # You can schedule periodic rebuilds by setting these env vars:
    #   CELERY_REBUILD_ENABLED=1
    #   CELERY_REBUILD_CRON="0 */6 * * *" (requires celery beat + crontab)
    # For simplicity, default is disabled.
    CELERY_REBUILD_ENABLED = os.getenv("CELERY_REBUILD_ENABLED", "0") == "1"

    if CELERY_REBUILD_ENABLED:
        from celery.schedules import crontab  # type: ignore

        # Defaults: daily at 03:00
        minute = int(os.getenv("CELERY_REBUILD_MINUTE", "0"))
        hour = int(os.getenv("CELERY_REBUILD_HOUR", "3"))

        # It can also be set which (yr, sub) pairs to rebuild:
        #   CELERY_REBUILD_PAIRS='[[5,1],[4,1]]'
        pairs = os.getenv("CELERY_REBUILD_PAIRS", "[]")
        try:
            rebuild_pairs = json.loads(pairs)
            if not isinstance(rebuild_pairs, list):
                rebuild_pairs = []
        except Exception:
            rebuild_pairs = []

        @cel.task(name="recommender.build_feature_store")
        def build_feature_store() -> dict[str, Any]:
            out: dict[str, Any] = {"ok": True, "rebuilt": []}
            for pair in rebuild_pairs:
                try:
                    yr, sub = int(pair[0]), int(pair[1])
                    meta = cache_rebuild(yr, sub)["meta"]
                    out["rebuilt"].append({"yr": yr, "sub": sub, **meta})
                except Exception as e:
                    out["rebuilt"].append({"pair": pair, "error": str(e)})
            return out

        cel.conf.beat_schedule = {
            "rebuild-feature-store": {
                "task": "recommender.build_feature_store",
                "schedule": crontab(minute=minute, hour=hour),
            }
        }

except Exception:
    # Celery is optional. If not installed/used, the rest of the module works.
    cel = None  # type: ignore





