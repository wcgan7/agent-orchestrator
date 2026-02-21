"""PRD ingestion/parsing helpers used by API routes."""

from __future__ import annotations

import re
from hashlib import sha256
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage.container import Container


def _normalize_prd_text(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n")


def _extract_task_candidates_from_chunk(chunk_text: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in chunk_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- ") or line.startswith("* "):
            title = re.sub(r"\s+", " ", line[2:].strip()).strip(" -")
            if len(title) >= 4:
                candidates.append(title)
                continue
        numbered = re.match(r"^\d+[\.\)]\s+(.*)$", line)
        if numbered:
            title = re.sub(r"\s+", " ", numbered.group(1).strip()).strip(" -")
            if len(title) >= 4:
                candidates.append(title)
                continue
        section_heading = re.match(r"^#{1,6}\s+(.*)$", line)
        if section_heading:
            title = re.sub(r"\s+", " ", section_heading.group(1).strip()).strip(" -")
            if len(title) >= 4:
                candidates.append(title)
                continue
    return candidates


def _fallback_chunk(text: str, chunk_size: int = 1200, overlap: int = 120) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    if not text.strip():
        return chunks
    idx = 0
    chunk_id = 1
    text_len = len(text)
    while idx < text_len:
        end = min(text_len, idx + chunk_size)
        chunk_text = text[idx:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "id": f"c{chunk_id}",
                    "strategy": "token_window",
                    "section_path": None,
                    "char_start": idx,
                    "char_end": end,
                    "text": chunk_text,
                }
            )
            chunk_id += 1
        if end >= text_len:
            break
        idx = max(end - overlap, idx + 1)
    return chunks


def _ingest_prd(content: str, default_priority: str) -> dict[str, Any]:
    normalized = _normalize_prd_text(content)
    lines = normalized.splitlines(keepends=True)

    chunks: list[dict[str, Any]] = []
    current_heading = "Document"
    current_block: list[str] = []
    block_start = 0
    cursor = 0
    chunk_id = 1
    heading_detected = False

    def flush_block(end_offset: int) -> None:
        nonlocal chunk_id, current_block, block_start
        block_text = "".join(current_block).strip()
        if not block_text:
            current_block = []
            block_start = end_offset
            return
        chunks.append(
            {
                "id": f"c{chunk_id}",
                "strategy": "heading_section",
                "section_path": current_heading,
                "char_start": block_start,
                "char_end": end_offset,
                "text": block_text,
            }
        )
        chunk_id += 1
        current_block = []
        block_start = end_offset

    for line in lines:
        stripped = line.strip()
        is_heading = bool(re.match(r"^#{1,6}\s+\S+", stripped))
        if is_heading:
            heading_detected = True
            flush_block(cursor)
            current_heading = re.sub(r"^#{1,6}\s+", "", stripped).strip() or "Document"
        current_block.append(line)
        cursor += len(line)
    flush_block(cursor)

    if not chunks:
        chunks = _fallback_chunk(normalized)
    elif not heading_detected:
        # Convert no-heading result into paragraph chunking when structure is flat.
        chunks = _fallback_chunk(normalized)
        for item in chunks:
            item["strategy"] = "paragraph_fallback"

    task_candidates: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for chunk in chunks:
        for title in _extract_task_candidates_from_chunk(str(chunk.get("text") or "")):
            key = title.lower().strip()
            if not key or key in seen_titles:
                continue
            seen_titles.add(key)
            task_candidates.append(
                {
                    "title": title[:200],
                    "priority": default_priority,
                    "source_chunk_id": chunk["id"],
                    "source_section_path": chunk.get("section_path"),
                }
            )

    ambiguity_warnings: list[str] = []
    if not task_candidates:
        ambiguity_warnings.append(
            "No explicit task bullets/headings found; generated generic task candidates from document chunks."
        )
        for chunk in chunks[:6]:
            chunk_text = str(chunk.get("text") or "").strip()
            first_sentence = re.split(r"(?<=[\.\!\?])\s+", chunk_text, maxsplit=1)[0].strip()
            if not first_sentence:
                continue
            title = re.sub(r"\s+", " ", first_sentence).strip(" -")
            if len(title) < 4:
                continue
            task_candidates.append(
                {
                    "title": title[:200],
                    "priority": default_priority,
                    "source_chunk_id": chunk["id"],
                    "source_section_path": chunk.get("section_path"),
                    "ambiguity_reason": "derived_from_first_sentence",
                }
            )

    if not task_candidates:
        task_candidates.append(
            {
                "title": "Imported PRD task",
                "priority": default_priority,
                "source_chunk_id": chunks[0]["id"] if chunks else None,
                "source_section_path": chunks[0].get("section_path") if chunks else None,
                "ambiguity_reason": "empty_or_unstructured_document",
            }
        )

    parsed_prd = {
        "strategy": chunks[0]["strategy"] if chunks else "empty",
        "chunk_count": len(chunks),
        "chunks": chunks,
        "task_candidates": task_candidates,
        "ambiguity_warnings": ambiguity_warnings,
    }
    return {
        "original_prd": {
            "content": content,
            "normalized_content": normalized,
            "char_count": len(normalized),
            "checksum_sha256": sha256(normalized.encode("utf-8")).hexdigest(),
        },
        "parsed_prd": parsed_prd,
    }


def _generated_tasks_from_parsed_prd(parsed_prd: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = parsed_prd.get("task_candidates")
    if not isinstance(candidates, list):
        return []
    out: list[dict[str, Any]] = []
    previous_id: str | None = None
    for idx, item in enumerate(candidates, start=1):
        if not isinstance(item, dict):
            continue
        generated_id = f"prd_{idx}"
        title = str(item.get("title") or f"Imported PRD task {idx}").strip() or f"Imported PRD task {idx}"
        priority = str(item.get("priority") or "P2").strip() or "P2"
        depends_on = [previous_id] if previous_id else []
        metadata = {
            "generated_ref_id": generated_id,
            "generated_depends_on": depends_on,
            "source_chunk_id": item.get("source_chunk_id"),
            "source_section_path": item.get("source_section_path"),
            "ingestion_source": "parsed_prd",
        }
        if item.get("ambiguity_reason"):
            metadata["ambiguity_reason"] = item.get("ambiguity_reason")
        out.append(
            {
                "id": generated_id,
                "title": title,
                "description": "",
                "task_type": "feature",
                "priority": priority,
                "depends_on": depends_on,
                "metadata": metadata,
            }
        )
        previous_id = generated_id
    return out


def _apply_generated_dep_links(container: Container, child_ids: list[str]) -> None:
    ref_to_task_id: dict[str, str] = {}
    for child_id in child_ids:
        child = container.tasks.get(child_id)
        if not child or not isinstance(child.metadata, dict):
            continue
        ref_id = str(child.metadata.get("generated_ref_id") or "").strip()
        if ref_id:
            ref_to_task_id[ref_id] = child.id

    for child_id in child_ids:
        child = container.tasks.get(child_id)
        if not child or not isinstance(child.metadata, dict):
            continue
        raw_deps = child.metadata.get("generated_depends_on")
        if not isinstance(raw_deps, list):
            continue
        changed = False
        for dep_ref in raw_deps:
            dep_id = ref_to_task_id.get(str(dep_ref or "").strip())
            if not dep_id or dep_id == child.id:
                continue
            if dep_id not in child.blocked_by:
                child.blocked_by.append(dep_id)
                changed = True
            dep_task = container.tasks.get(dep_id)
            if dep_task and child.id not in dep_task.blocks:
                dep_task.blocks.append(child.id)
                container.tasks.upsert(dep_task)
        if changed:
            container.tasks.upsert(child)
