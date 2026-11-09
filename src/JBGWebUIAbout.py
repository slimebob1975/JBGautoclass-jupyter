"""About tab of the web interface (revision 295d).

What the program does and how to use it, from the user guide in the
repository (``docs/user_guide.sv.md`` and ``docs/user_guide.en.md``; the
README is developer-oriented), plus the program's version line.

The guide is read on every request, as the Swedish catalogue is, so an
updated guide needs no server restart. It is shown through a small Markdown
subset (headings, paragraphs, lists, quotes, bold, code, links) that escapes
all text first: no HTML from the file reaches the page, and links go only to
the interface itself, to anchors on the page or to https addresses. No new
dependency.
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import Depends, Request
from fastapi.responses import HTMLResponse
from markupsafe import Markup, escape

GUIDES = Path(__file__).resolve().parents[1] / "docs"
GUIDE_LANGUAGES = ("sv", "en")

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_BULLET = re.compile(r"^[-*]\s+(.*)$")
_NUMBER = re.compile(r"^\d+[.)]\s+(.*)$")
_QUOTE = re.compile(r"^>\s?(.*)$")
_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_HEADING_TAGS = {1: "h2", 2: "h3", 3: "h4"}     # the page's own title is the card's


def load_guide(language: str, root: Path | None = None) -> tuple[str | None, str | None]:
    """The guide in ``language``, else in the other language; (text, language)."""
    root = GUIDES if root is None else Path(root)
    order = (language,) + tuple(other for other in GUIDE_LANGUAGES if other != language)
    for candidate in order:
        if candidate not in GUIDE_LANGUAGES:
            continue
        try:
            return (root / f"user_guide.{candidate}.md").read_text(encoding="utf-8"), candidate
        except (OSError, UnicodeDecodeError):
            continue
    return None, None


def _safe_target(target: str) -> bool:
    if target.startswith("#"):
        return len(target) > 1
    if target.startswith("/ui/") or target == "/ui":
        return "//" not in target and "\\" not in target
    return target.startswith("https://")


def _bold(escaped: str) -> str:
    return _BOLD.sub(r"<strong>\1</strong>", escaped)


def _links(text: str) -> str:
    out, position = [], 0
    for match in _LINK.finditer(text):
        out.append(_bold(str(escape(text[position:match.start()]))))
        label, target = _bold(str(escape(match.group(1)))), match.group(2)
        if _safe_target(target):
            external = ' rel="noopener noreferrer"' if target.startswith("https://") else ""
            out.append(f'<a href="{escape(target)}"{external}>{label}</a>')
        else:
            out.append(label)
        position = match.end()
    out.append(_bold(str(escape(text[position:]))))
    return "".join(out)


def _inline(text: str) -> str:
    """Code spans first (their content is shown as written), then links and bold."""
    out, position = [], 0
    for match in _CODE.finditer(text):
        out.append(_links(text[position:match.start()]))
        out.append(f"<code>{escape(match.group(1))}</code>")
        position = match.end()
    out.append(_links(text[position:]))
    return "".join(out)


def _slug(title: str, used: set[str]) -> str:
    base = re.sub(r"[^\w]+", "-", title.lower()).strip("-_") or "section"
    slug, number = base, 2
    while slug in used:
        slug, number = f"{base}-{number}", number + 1
    used.add(slug)
    return slug


def render_markdown(text: str) -> tuple[Markup, list[tuple[str, str]]]:
    """HTML for the supported Markdown subset, and the contents (``##`` headings)."""
    blocks: list[str] = []
    contents: list[tuple[str, str]] = []
    used: set[str] = set()
    kind: str | None = None
    lines: list[str] = []

    def flush():
        nonlocal kind, lines
        if kind == "p":
            blocks.append(f"<p>{_inline(' '.join(lines))}</p>")
        elif kind in ("ul", "ol"):
            items = "".join(f"<li>{_inline(item)}</li>" for item in lines)
            blocks.append(f"<{kind}>{items}</{kind}>")
        elif kind == "quote":
            blocks.append(f"<blockquote><p>{_inline(' '.join(lines))}</p></blockquote>")
        kind, lines = None, []

    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        heading = _HEADING.match(stripped) if not raw[:1].isspace() else None
        bullet, number = _BULLET.match(stripped), _NUMBER.match(stripped)
        quote = _QUOTE.match(stripped)
        if heading:
            flush()
            level, title = len(heading.group(1)), heading.group(2)
            slug = _slug(title, used)
            tag = _HEADING_TAGS[level]
            blocks.append(f'<{tag} id="{slug}">{_inline(title)}</{tag}>')
            if level == 2:
                contents.append((slug, title))
        elif bullet or number:
            wanted = "ul" if bullet else "ol"
            if kind != wanted:
                flush()
                kind = wanted
            lines.append((bullet or number).group(1))
        elif quote:
            if kind != "quote":
                flush()
                kind = "quote"
            lines.append(quote.group(1))
        elif kind in ("ul", "ol"):
            lines[-1] = f"{lines[-1]} {stripped}"        # continuation of the item
        else:
            if kind not in ("p", "quote"):
                flush()
                kind = "p"
            lines.append(stripped)
    flush()
    return Markup("\n".join(blocks)), contents


def add_about_tab(app, *, principal_dependency, render) -> None:
    from JBGWebUI import request_language

    @app.get("/ui/about", response_class=HTMLResponse, include_in_schema=False)
    def about_tab(request: Request, principal=Depends(principal_dependency)):
        text, shown_language = load_guide(request_language(request))
        guide, contents = render_markdown(text) if text is not None else (None, [])
        return render(request, principal, "about.html", "about", guide=guide,
                      contents=contents, guide_language=shown_language)
