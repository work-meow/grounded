"""Whether an address somebody typed is one this server may connect to.

Every connector until now talked to a hardcoded host — api.notion.com,
cloud-api.yandex.net, graph.microsoft.com. An S3-compatible store is the first
where the *user* names the machine, and that is a different thing entirely: the
indexer runs on a private docker network alongside the object store, the API,
the database and whatever else the box hosts. A form field that reaches
``http://minio:9000``, ``http://127.0.0.1:8000`` or the cloud metadata service
at 169.254.169.254 is not a typo, it is a way for anyone holding a token to
make this server fetch things on their behalf.

So the rule is the plain one: the host must resolve, and every address it
resolves to must be a public one. Not a blocklist of names — those are trivially
worked around, by an IP in decimal, by a hostname that points at 127.0.0.1, by
IPv6 — but a check on the addresses themselves, after resolution, which is where
all of those end up looking the same.
"""

import ipaddress
import socket
from urllib.parse import urlsplit

#: Anything else is not something to fetch over. `file:`, `gopher:` and the rest
#: are why this is a list of what is allowed rather than of what is not.
SCHEMES = ("http", "https")


def verify_public(url: str) -> None:
    """Raise ValueError with a readable reason unless this URL is safe to fetch.

    ponytail: a host is checked here and connected to later, by a client that
    resolves the name again — so a name that answers with a public address now
    and a private one a moment later gets through. Ceiling: DNS rebinding by
    somebody who already holds a token on this deployment. Upgrade path: pin the
    resolved address into the connection, which means terminating TLS against an
    IP while validating the certificate for the name — neither botocore nor
    httpx exposes that without replacing the transport. The check is repeated
    before every poll, so the window is one request rather than for ever.
    """
    parts = urlsplit(url.strip())
    if parts.scheme not in SCHEMES:
        raise ValueError("адрес должен начинаться с http:// или https://")
    host = parts.hostname
    if not host:
        raise ValueError("в адресе нет имени сервера")

    for address in _addresses(host):
        if not address.is_global:
            raise ValueError(
                f"адрес {host} ведёт внутрь сети ({address}) — так подключаться нельзя"
            )


def _addresses(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Every address this host answers with. Raises ValueError if it answers none.

    A literal address goes through the same resolver, which is what makes the
    exotic spellings — 2130706433, 0x7f000001, ::ffff:127.0.0.1 — come out as
    the addresses they are rather than as strings to compare.
    """
    try:
        found = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"не удалось определить адрес {host}") from exc

    addresses = []
    for *_, sockaddr in found:
        try:
            addresses.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:  # pragma: no cover - getaddrinfo does not return these
            continue
    if not addresses:
        raise ValueError(f"не удалось определить адрес {host}")
    return addresses
