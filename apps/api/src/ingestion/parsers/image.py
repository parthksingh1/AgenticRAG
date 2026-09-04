"""Image parsing via OCR, with an optional vision-model pass.

Two tiers, because they fail in opposite ways:

* **Tesseract** is fast, free and offline, and it is good at clean printed text.
  It is hopeless at layout: a table comes back as a jumble of numbers with the
  column structure destroyed.
* **A vision model** reads layout correctly, including tables and handwriting,
  but costs money per image and can hallucinate text that is not there.

So OCR runs first, and the vision pass is opt-in per tenant and used where its
strength matters — complex tables and diagrams. Vision output is recorded with
``extractor: vision`` in the block metadata so that a later investigation can
tell which text a model read versus transcribed.

Example:
    >>> ImageParser().formats == frozenset({"image"})
    True
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

from src.core.logging import get_logger
from src.ingestion.parsers.base import Parser, register_parser
from src.ingestion.parsers.text import _title_from_filename
from src.ingestion.types import ParsedDocument, TextBlock
from src.models.document import ChunkKind

if TYPE_CHECKING:
    from src.services.llm.router import LLMRouter

log = get_logger(__name__)

#: Below this many OCR characters the image is treated as non-textual (a photo,
#: a logo, a chart) and, if enabled, escalated to the vision model.
VISION_ESCALATION_CHARS = 200

_VISION_PROMPT = (
    "Transcribe all text in this image exactly as it appears. Render any table as "
    "a Markdown table, preserving rows and columns. If the image contains a chart, "
    "describe its axes and the values shown. Transcribe only what is visible; do "
    "not infer or complete missing text."
)


class ImageParser(Parser):
    """Reads text out of images."""

    name = "pytesseract"
    formats = frozenset({"image"})

    def __init__(
        self,
        *,
        router: LLMRouter | None = None,
        vision_model: str | None = None,
        language: str = "eng",
    ) -> None:
        """Create the parser.

        Args:
            router: Router used for the vision pass. ``None`` disables it.
            vision_model: Vision-capable model id.
            language: Tesseract language pack.
        """
        self._router = router
        self._vision_model = vision_model
        self._language = language

    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument:
        """OCR an image into text blocks."""
        text = self._ocr(data)
        blocks = [
            TextBlock(text=paragraph.strip(), page_number=1, metadata={"extractor": "ocr"})
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        return ParsedDocument(
            blocks=tuple(blocks),
            title=_title_from_filename(filename),
            parser=self.name,
            page_count=1,
            metadata={
                "ocr_chars": len(text),
                "vision_recommended": len(text) < VISION_ESCALATION_CHARS,
            },
        )

    async def parse_with_vision(
        self, data: bytes, *, filename: str | None = None, mime_type: str = "image/png"
    ) -> ParsedDocument:
        """Parse using OCR, escalating to the vision model when OCR is thin.

        Kept separate from :meth:`parse` because it is async and costs money;
        the synchronous path stays usable from the Celery worker without pulling
        an event loop into it.
        """
        ocr_document = self.parse(data, filename=filename)
        thin = ocr_document.char_length < VISION_ESCALATION_CHARS
        if not (thin and self._router and self._vision_model):
            return ocr_document

        transcription = await self._vision_transcribe(data, mime_type=mime_type)
        if not transcription:
            return ocr_document

        blocks = tuple(
            TextBlock(
                text=part.strip(),
                kind=ChunkKind.TABLE if part.lstrip().startswith("|") else ChunkKind.PROSE,
                page_number=1,
                metadata={"extractor": "vision", "model": self._vision_model},
            )
            for part in transcription.split("\n\n")
            if part.strip()
        )
        return ParsedDocument(
            blocks=blocks,
            title=ocr_document.title,
            parser=f"{self.name}+vision",
            page_count=1,
            metadata={"extractor": "vision", "ocr_chars": ocr_document.char_length},
        )

    # ── internals ────────────────────────────────────────────────────────────

    def _ocr(self, data: bytes) -> str:
        """Run Tesseract, returning an empty string when it is unavailable."""
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            log.warning("OCR dependencies unavailable; image will yield no text")
            return ""

        try:
            with Image.open(io.BytesIO(data)) as image:
                return str(
                    pytesseract.image_to_string(image.convert("RGB"), lang=self._language)
                ).strip()
        except Exception as exc:  # noqa: BLE001 - a bad image must not fail ingestion
            log.warning("OCR failed", reason=str(exc))
            return ""

    async def _vision_transcribe(self, data: bytes, *, mime_type: str) -> str | None:
        """Ask the vision model to transcribe the image."""
        if self._router is None or self._vision_model is None:
            return None

        from src.services.llm.types import CompletionRequest, Message

        encoded = base64.b64encode(data).decode("ascii")
        request = CompletionRequest(
            messages=(
                Message.user(f"{_VISION_PROMPT}\n\n[image:{mime_type};base64,{encoded[:64]}...]"),
            ),
            model=self._vision_model,
            max_tokens=2048,
            temperature=0.0,
            node="vision_ocr",
        )
        try:
            completion = await self._router.complete(request, allow_fallback=False)
        except Exception as exc:  # noqa: BLE001 - vision is an optional enhancement
            log.warning("vision transcription failed; keeping OCR output", reason=str(exc))
            return None
        return completion.content.strip() or None


image_parser = register_parser(ImageParser())
