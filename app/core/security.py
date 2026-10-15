from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def redact_token(token: str | None) -> str:
    """
    Redact a token for logging purposes.
    Shows the first 6 characters followed by ***.
    """
    if not token:
        return "None"
    if len(token) <= 6:
        return token
    return f"{token[:6]}***"


def redact_url_query(url: object) -> str:
    """Return a log-safe URL with every query-string value redacted.

    Query parameter names remain visible for diagnostics, but values never do.
    Redacting every value avoids maintaining an incomplete list of possible
    secret names such as api_key, token, access_token, client_secret, etc.
    """
    text = str(url)

    try:
        parts = urlsplit(text)
    except ValueError:
        if "?" in text:
            return f"{text.split('?', 1)[0]}?[REDACTED]"
        return text

    if not parts.query:
        return text

    items = parse_qsl(
        parts.query,
        keep_blank_values=True,
    )

    if items:
        safe_query = urlencode(
            [
                (key, "[REDACTED]")
                for key, _value in items
            ],
            doseq=True,
        )
    else:
        safe_query = "[REDACTED]"

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            safe_query,
            parts.fragment,
        )
    )
