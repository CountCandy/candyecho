"""Extract plain text from uploaded documents (.txt, .epub).

Uses only the Python standard library (zipfile + html.parser + ElementTree),
so it adds no dependencies — important because the pinned cu130 torch index
makes regenerating the lockfile awkward in some environments.
"""
import io
import re
import zipfile
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

_CONTAINER_NS = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}


def _decode_text(data: bytes) -> str:
    """Decode bytes to str, trying the most common text encodings in turn."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


class _HTMLTextParser(HTMLParser):
    """Collapse HTML into readable text, inserting breaks at block elements."""

    _BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "blockquote"}
    _DROP = {"script", "style", "head", "title"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._DROP:
            self._skip += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._DROP and self._skip:
            self._skip -= 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _strip_html(html: str) -> str:
    parser = _HTMLTextParser()
    try:
        parser.feed(html)
    except Exception:
        return ""
    return parser.get_text()


def _localname(tag: str) -> str:
    """Return an XML tag's local name, dropping any {namespace} prefix."""
    return tag.rsplit("}", 1)[-1]


def _spine_documents(z: zipfile.ZipFile, names: list[str]) -> list[str]:
    """Return the .xhtml documents in reading order (spine), if resolvable."""
    try:
        container = z.read("META-INF/container.xml")
        root = ET.fromstring(container)
        rootfile = root.find(".//c:rootfile", _CONTAINER_NS)
        opf_path = rootfile.get("full-path") if rootfile is not None else None
    except Exception:
        opf_path = None

    if not opf_path or opf_path not in names:
        return []

    try:
        opf_root = ET.fromstring(z.read(opf_path))
    except Exception:
        return []

    base = opf_path.rsplit("/", 1)[0] if "/" in opf_path else ""
    manifest: dict[str, str] = {}
    for el in opf_root.iter():
        if _localname(el.tag) == "item":
            iid, href = el.get("id"), el.get("href")
            if iid and href:
                manifest[iid] = href

    docs: list[str] = []
    for el in opf_root.iter():
        if _localname(el.tag) == "itemref":
            href = manifest.get(el.get("idref"))
            if not href:
                continue
            full = (base + "/" + href) if base else href
            full = re.sub(r"/{2,}", "/", full)
            if full in names:
                docs.append(full)
    return docs


def extract_epub_text(data: bytes) -> str:
    """Extract reading-order text from an .epub (a ZIP of XHTML documents)."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        docs = _spine_documents(z, names)
        if not docs:
            # Fallback: every (x)html document, in archive order.
            docs = [n for n in names if n.lower().endswith((".xhtml", ".html", ".htm"))]

        chunks: list[str] = []
        for doc in docs:
            try:
                text = _strip_html(_decode_text(z.read(doc))).strip()
            except Exception:
                continue
            if text:
                chunks.append(text)

    text = "\n\n".join(chunks)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_upload(filename: str, data: bytes) -> str:
    """Extract plain text from an uploaded .txt or .epub file's bytes."""
    if (filename or "").lower().endswith(".epub"):
        return extract_epub_text(data)
    return _decode_text(data).strip()
