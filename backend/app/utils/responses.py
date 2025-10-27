"""Request parsing helpers."""
from __future__ import annotations

from flask import Request


def parse_request(req: Request) -> dict:
    if not req.is_json:
        raise ValueError("Expected JSON request body")
    payload = req.get_json(silent=True)
    if payload is None:
        raise ValueError("Malformed JSON payload")
    return payload
