# """
# modules/word_renderer.py
# Renders source context for Word (.docx) files with text highlighting.
# This is the Word equivalent of PDF eye-popup traceability.
# """

# from __future__ import annotations

# import html
# import re

# from modules.word_parser import extract_word_blocks


# def _highlight_text(text: str, needle: str) -> str:
#     """
#     Safely highlight occurrences of needle inside text using <mark>.
#     """
#     if not text or not needle:
#         return html.escape(text or "")

#     escaped_text = html.escape(text)
#     escaped_needle = html.escape(needle)

#     pattern = re.compile(re.escape(escaped_needle), re.IGNORECASE)
#     return pattern.sub(lambda m: f"<mark style='background:#fde047;padding:0 2px;border-radius:3px;'>{m.group(0)}</mark>", escaped_text)


# def render_word_context_with_highlight(
#     file_path: str,
#     search_text: str,
#     source_block: int | None = None,
#     context_radius: int = 1,
# ) -> str:
#     """
#     Return HTML snippet showing the source paragraph / table row with highlighted match.

#     Strategy:
#       1. If source_block is provided, show that block and nearby context.
#       2. Otherwise search all blocks for the first match.
#     """
#     blocks = extract_word_blocks(file_path)
#     if not blocks:
#         return (
#             "<div style='padding:12px;border:1px solid #334155;border-radius:10px;'>"
#             "No readable content found in Word document."
#             "</div>"
#         )

#     hit_idx = None

#     if source_block is not None:
#         for i, b in enumerate(blocks):
#             if b.get("block_id") == source_block:
#                 hit_idx = i
#                 break

#     if hit_idx is None and search_text:
#         for i, b in enumerate(blocks):
#             if search_text.lower() in (b.get("text", "").lower()):
#                 hit_idx = i
#                 break

#     if hit_idx is None:
#         hit_idx = 0

#     start = max(0, hit_idx - context_radius)
#     end = min(len(blocks), hit_idx + context_radius + 1)
#     snippet_blocks = blocks[start:end]

#     html_parts = [
#         "<div style='padding:12px;border:1px solid #334155;border-radius:12px;background:#0b1220;'>",
#         "<div style='font-size:12px;color:#94a3b8;margin-bottom:10px;'>📄 WORD source context</div>"
#     ]

#     for i, b in enumerate(snippet_blocks):
#         block_type = b.get("block_type", "block")
#         label = block_type.replace("_", " ").title()

#         txt = b.get("text", "")
#         if (start + i) == hit_idx:
#             rendered = _highlight_text(txt, search_text)
#             border = "2px solid #facc15"
#             bg = "#111827"
#         else:
#             rendered = html.escape(txt)
#             border = "1px solid #1f2937"
#             bg = "#0f172a"

#         meta_bits = []
#         if b.get("para_index") is not None:
#             meta_bits.append(f"Paragraph {b['para_index'] + 1}")
#         if b.get("table_index") is not None:
#             meta_bits.append(f"Table {b['table_index'] + 1}")
#         if b.get("row_index") is not None:
#             meta_bits.append(f"Row {b['row_index'] + 1}")

#         meta = " · ".join(meta_bits) if meta_bits else label

#         html_parts.append(
#             f"""
#             <div style="margin-bottom:10px;padding:10px;border:{border};border-radius:10px;background:{bg};">
#                 <div style="font-size:11px;color:#94a3b8;margin-bottom:6px;">{html.escape(meta)}</div>
#                 <div style="font-size:14px;line-height:1.55;color:#e5e7eb;white-space:pre-wrap;">{rendered}</div>
#             </div>
#             """
#         )

#     html_parts.append("</div>")
#     return "".join(html_parts)

"""
modules/word_renderer.py
Renders source context for Word (.docx) files with text highlighting.
"""

from __future__ import annotations

import html
import re

from modules.word_parser import extract_word_blocks


def _highlight_text(text: str, needle: str) -> str:
    """
    Highlight occurrences of needle safely inside already-escaped HTML text.
    """
    if not text:
        return ""

    escaped_text = html.escape(text)

    if not needle:
        return escaped_text

    escaped_needle = html.escape(needle.strip())
    if not escaped_needle:
        return escaped_text

    pattern = re.compile(re.escape(escaped_needle), re.IGNORECASE)
    return pattern.sub(
        lambda m: f"<mark style='background:#fde047;color:#111827;padding:0 2px;border-radius:3px;'>{m.group(0)}</mark>",
        escaped_text,
    )


def render_word_context_with_highlight(
    file_path: str,
    search_text: str,
    source_block: int | None = None,
    context_radius: int = 1,
) -> str:
    """
    Return HTML snippet showing the source paragraph / table row with highlighted match.
    """
    blocks = extract_word_blocks(file_path)
    if not blocks:
        return """
        <div style="padding:12px;border:1px solid #334155;border-radius:10px;background:#0b1220;color:#e5e7eb;">
            No readable content found in Word document.
        </div>
        """

    hit_idx = None

    if source_block is not None:
        for i, b in enumerate(blocks):
            if b.get("block_id") == source_block:
                hit_idx = i
                break

    if hit_idx is None and search_text:
        for i, b in enumerate(blocks):
            if search_text.lower() in (b.get("text", "").lower()):
                hit_idx = i
                break

    if hit_idx is None:
        hit_idx = 0

    start = max(0, hit_idx - context_radius)
    end = min(len(blocks), hit_idx + context_radius + 1)
    snippet_blocks = blocks[start:end]

    html_parts = [
        "<div style='padding:12px;border:1px solid #334155;border-radius:12px;background:#0b1220;'>",
        "<div style='font-size:12px;color:#94a3b8;margin-bottom:10px;'>📄 WORD source context</div>"
    ]

    for i, b in enumerate(snippet_blocks):
        txt = b.get("text", "")
        is_hit = (start + i) == hit_idx

        rendered = _highlight_text(txt, search_text) if is_hit else html.escape(txt)

        meta_bits = []
        if b.get("para_index") is not None:
            meta_bits.append(f"Paragraph {b['para_index'] + 1}")
        if b.get("table_index") is not None:
            meta_bits.append(f"Table {b['table_index'] + 1}")
        if b.get("row_index") is not None:
            meta_bits.append(f"Row {b['row_index'] + 1}")

        meta = " · ".join(meta_bits) if meta_bits else b.get("block_type", "Block").replace("_", " ").title()

        border = "2px solid #facc15" if is_hit else "1px solid #1f2937"
        bg = "#111827" if is_hit else "#0f172a"

        html_parts.append(
            f"""
            <div style="margin-bottom:10px;padding:10px;border:{border};border-radius:10px;background:{bg};">
                <div style="font-size:11px;color:#94a3b8;margin-bottom:6px;">{meta}</div>
                <div style="font-size:14px;line-height:1.55;color:#e5e7eb;white-space:pre-wrap;">{rendered}</div>
            </div>
            """
        )

    html_parts.append("</div>")
    return "".join(html_parts)