"""The HTTP behaviour the polled connectors share.

Four services, four APIs, one set of things that go wrong: a rate limit, a bad
gateway, a connection dropped mid-listing. Handling that once here keeps each
connector down to the part that is actually specific to it.

Retries are conservative on purpose. Every request these connectors make is a
read — a listing or a download — so retrying one cannot duplicate anything, but
retrying hard against a service that is rate-limiting us is how an integration
gets suspended. Three attempts, honouring ``Retry-After`` when the service sends
one, and never more than a minute of waiting in total.
"""

import logging
import time
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_ATTEMPTS = 3
_MAX_BACKOFF_S = 20.0
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class Http:
    """A synchronous client. Lives on the connector's polling thread."""

    __slots__ = ("_client", "_label")

    def __init__(self, label: str, *, headers: dict[str, str] | None = None, timeout: float = 60.0):
        self._label = label
        self._client = httpx.Client(
            headers=headers or {},
            timeout=timeout,
            # Downloads redirect to a CDN on three of the four services.
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def set_header(self, name: str, value: str) -> None:
        self._client.headers[name] = value

    def json(self, method: str, url: str, **kwargs: Any) -> Any:
        return self.request(method, url, **kwargs).json()

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """A response with a 2xx status, or the last exception raised trying."""
        for attempt in range(1, _ATTEMPTS + 1):
            last = attempt == _ATTEMPTS
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.HTTPError:
                if last:
                    raise
                self._wait(attempt, None)
                continue

            if response.status_code in _RETRY_STATUSES and not last:
                self._wait(attempt, response.headers.get("Retry-After"))
                continue
            # Raises for 4xx, and for a 5xx that outlived the retries. The body
            # is not logged: for these services it can echo the request, and the
            # request carries the credential.
            response.raise_for_status()
            return response
        raise AssertionError("unreachable")  # pragma: no cover

    def _wait(self, attempt: int, retry_after: str | None) -> None:
        delay = min(2.0**attempt, _MAX_BACKOFF_S)
        if retry_after:
            try:
                delay = min(float(retry_after), _MAX_BACKOFF_S)
            except ValueError:  # an HTTP-date rather than seconds; the default is fine
                pass
        logger.info("%s: retrying in %.0fs (attempt %d)", self._label, delay, attempt)
        time.sleep(delay)


def as_timestamp(value: str | None) -> int:
    """An ISO-8601 instant as unix seconds, or 0 when the service omitted it.

    0 rather than "now": this is the change detector, and a value that moves on
    every poll would re-download and re-embed the document on every poll.
    """
    if not value:
        return 0
    try:
        # Python parses "+00:00" but not the "Z" every one of these APIs sends.
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


class OAuthToken:
    """An access token that expires, and the refresh token that renews it.

    Dropbox and Microsoft both issue access tokens good for about an hour, so
    what the user pastes is a refresh token alongside the application's own
    identity. The connector trades those for a fresh access token whenever the
    one it holds is near expiry, which means a connected source keeps working
    for as long as the user leaves it connected and there is nothing here to
    rotate on a schedule.

    Refreshing goes through its own client: the connector's client carries the
    ``Authorization`` header of the token being replaced, and sending an expired
    bearer to a token endpoint is how you get a confusing 400.
    """

    __slots__ = ("_auth", "_data", "_expires_at", "_http", "_token", "_url")

    #: Renew this many seconds early, so a token cannot expire between the check
    #: and the request it was fetched for.
    _SKEW_S = 120.0

    def __init__(
        self,
        label: str,
        *,
        url: str,
        data: dict[str, str],
        auth: tuple[str, str] | None = None,
    ) -> None:
        self._http = Http(f"{label} token", timeout=30.0)
        self._url = url
        self._data = data
        self._auth = auth
        self._token = ""
        self._expires_at = 0.0

    def close(self) -> None:
        self._http.close()

    def header(self) -> str:
        """``Bearer <token>``, refreshing first if the current one is stale."""
        if time.monotonic() >= self._expires_at:
            payload = self._http.json("POST", self._url, data=self._data, auth=self._auth)
            self._token = payload["access_token"]
            # Services are allowed to omit expires_in; an hour is what both of
            # ours document, and being early costs one extra request.
            lifetime = float(payload.get("expires_in") or 3600)
            self._expires_at = time.monotonic() + max(lifetime - self._SKEW_S, 60.0)
        return f"Bearer {self._token}"
