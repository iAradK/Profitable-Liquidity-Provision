"""Thread-safe GraphQL client for The Graph gateway, with a shared rate limit and retries."""

import logging
import random
import threading
import time

import requests

from .config import NetworkConfig, SubgraphConfig

log = logging.getLogger("subgraph")

MAX_BACKOFF_SECONDS = 120
TRANSIENT_HINTS = ("timeout", "timed out", "indexer", "unavailable", "too many", "rate limit",
                   "try again", "internal", "bad gateway", "not caught up", "store error")


class GraphQLError(Exception):
    """The query was rejected (bad field, bad API key...); retrying will not help."""


class TransientGraphQLError(GraphQLError):
    """The gateway or indexer had a temporary problem; worth retrying."""


class GraphClient:
    def __init__(self, subgraph: SubgraphConfig, network: NetworkConfig):
        self._url = subgraph.url
        self._network = network
        self._local = threading.local()  # one requests.Session per worker thread
        self._lock = threading.Lock()
        self._last_request = 0.0
        self.n_requests = 0

    def query(self, query: str) -> dict:
        """POST a query; retries with exponential backoff on network errors, HTTP 429/5xx and
        transient GraphQL errors. Permanent errors raise GraphQLError immediately."""
        max_retries = self._network.max_retries
        for attempt in range(max_retries + 1):
            try:
                return self._post(query)
            except (requests.RequestException, TransientGraphQLError) as exc:
                if attempt == max_retries:
                    raise
                delay = min(MAX_BACKOFF_SECONDS, 2 ** attempt) * (0.5 + random.random())
                log.warning("request failed (%s), retry %d/%d in %.1fs",
                            str(exc)[:200], attempt + 1, max_retries, delay)
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _post(self, query: str) -> dict:
        self._wait_for_turn()
        response = self._session().post(self._url, json={"query": query},
                                        timeout=self._network.timeout_seconds)
        if response.status_code == 429 or response.status_code >= 500:
            raise requests.HTTPError(f"HTTP {response.status_code}: {response.text[:200]}")
        if response.status_code >= 400:
            raise GraphQLError(f"HTTP {response.status_code}: {response.text[:300]}")
        body = response.json()
        errors = body.get("errors")
        if errors:
            message = "; ".join(str(e.get("message", e)) for e in errors)
            if any(hint in message.lower() for hint in TRANSIENT_HINTS):
                raise TransientGraphQLError(message[:300])
            raise GraphQLError(message)
        return body["data"]

    def _wait_for_turn(self) -> None:
        """Keep at least min_interval_seconds between requests across all threads."""
        with self._lock:
            wait = self._last_request + self._network.min_interval_seconds - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            self.n_requests += 1

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session
