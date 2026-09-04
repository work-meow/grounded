"""What this system can read.

One list, because three places need the same answer and they are in different
processes: the API rejects an upload it could not index, the connectors skip a
remote file they could not index, and the indexer's parser dispatches on it.

Kept as a suffix map rather than a mime-type map because that is the only handle
every one of those places has in common — an uploaded filename, a Drive entry,
a Dropbox path. The parser itself sniffs magic bytes and never trusts this.
"""

from pathlib import PurePosixPath

#: Suffix to the mime type we record for it. This is what the API stores and
#: what a browser is told when it opens the original, so it must be the real
#: type, not a guess from the client's Content-Type header.
SUFFIXES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".txt": "text/plain",
    ".md": "text/markdown",
}

#: For an error message, in the order a person would read them.
HUMAN_READABLE = ", ".join(sorted(SUFFIXES))


def mime_for(filename: str) -> str | None:
    """The mime type for a filename, or None if we cannot read that format."""
    return SUFFIXES.get(PurePosixPath(filename).suffix.lower())


def is_supported(filename: str) -> bool:
    return mime_for(filename) is not None
