"""Portable BF16 bit-pattern snapshots and bounded sparse replacements (no pickle)."""
from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .contracts import JSON, canonical, digest, file_sha, inside, integer, items, loads, sha, string

MAX_PATCH_BYTES = 16 * 1024 * 1024
MAX_CHANGED = 262144
MAX_SNAPSHOT_BYTES = 2 * 1024**3


def floats(bits: NDArray[np.uint16]) -> NDArray[np.float32]:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def bf16(values: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Round float32 to BF16, ties to even; reject nonfinite before and after."""
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("finite float32 required")
    raw = values.view(np.uint32)
    result = ((raw + np.uint32(0x7FFF) + ((raw >> 16) & 1)) >> 16).astype(np.uint16)
    if not np.isfinite(floats(result)).all():
        raise ValueError("BF16 overflow")
    return result


def metadata(document: dict[str, JSON]) -> dict[str, dict[str, JSON]]:
    if document.get("schema") != "WSPTensors/1":
        raise ValueError("expected WSPTensors/1")
    result = {}
    total = 0
    for tensor in items(document.get("tensors")):
        name = string(tensor.get("name"))
        shape = tensor.get("shape")
        if not isinstance(shape, list) or not shape or any(integer(n) <= 0 for n in shape):
            raise ValueError("invalid tensor shape")
        if name in result or tensor.get("dtype") != "bfloat16":
            raise ValueError("duplicate tensor or non-BF16 checkpoint")
        digest(tensor.get("sha256"))
        total += math.prod(integer(n) for n in shape) * 2
        result[name] = tensor
    if not result or len(result) > 32 or total > MAX_SNAPSHOT_BYTES:
        raise ValueError("tensor allowlist exceeds bounded snapshot budget")
    return result


def validate_patch(data: bytes, document: dict[str, JSON], snapshots: Path | None = None
                   ) -> tuple[list[tuple[str, NDArray[np.int64], NDArray[np.uint16]]], dict[str, JSON]]:
    allowed = metadata(document)
    if len(data) > MAX_PATCH_BYTES:
        raise ValueError("patch byte budget exceeded")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or sum(x.file_size for x in archive.infolist()) > MAX_PATCH_BYTES:
            raise ValueError("duplicate members or decompressed patch budget exceeded")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if "meta" not in archive.files or archive["meta"].dtype != np.uint8 or archive["meta"].ndim != 1:
            raise ValueError("missing byte metadata")
        meta = loads(archive["meta"].tobytes().decode("utf-8"))
        updates = items(meta.get("updates"))
        if meta.get("schema") != "WSPPatch/1" or not updates or len(updates) > len(allowed):
            raise ValueError("invalid patch metadata")
        expected = {"meta"} | {f"{p}{n}" for n in range(len(updates)) for p in ("i", "v")}
        if set(archive.files) != expected:
            raise ValueError("unexpected patch members")
        seen: set[str] = set()
        result = []
        receipts: list[JSON] = []
        count = 0
        for n, entry in enumerate(updates):
            name = string(entry.get("name"))
            if name not in allowed or name in seen:
                raise ValueError("tensor not allowlisted or repeated")
            seen.add(name)
            spec = allowed[name]
            if entry.get("shape") != spec["shape"] or entry.get("dtype") != spec["dtype"]:
                raise ValueError("shape/dtype mismatch")
            if digest(entry.get("base_sha256")) != spec["sha256"]:
                raise ValueError("base tensor hash mismatch")
            indices, values = archive[f"i{n}"], archive[f"v{n}"]
            if (indices.dtype != np.int64 or values.dtype != np.uint16 or indices.ndim != 1 or
                    values.ndim != 1 or not len(indices) or len(indices) != len(values)):
                raise ValueError("indices int64 and BF16 uint16 values must be matching nonempty vectors")
            count += len(indices)
            shape = spec["shape"]
            if not isinstance(shape, list):
                raise ValueError("invalid shape")
            size = math.prod(integer(v) for v in shape)
            if count > MAX_CHANGED or indices[0] < 0 or indices[-1] >= size or np.any(np.diff(indices) <= 0):
                raise ValueError("index or changed-entry budget violation")
            if not np.isfinite(floats(values)).all():
                raise ValueError("nonfinite tensor update")
            update_hash = sha(indices.tobytes() + values.tobytes())
            if digest(entry.get("update_sha256")) != update_hash:
                raise ValueError("update hash mismatch")
            if snapshots is not None:
                path = inside(snapshots, spec.get("file"))
                if file_sha(path) != digest(spec.get("file_sha256")):
                    raise ValueError("snapshot file hash mismatch")
                base = np.load(path, allow_pickle=False)
                if base.dtype != np.uint16 or list(base.shape) != shape or sha(base.tobytes()) != spec["sha256"]:
                    raise ValueError("snapshot tensor hash/shape/dtype mismatch")
                if np.any(base.reshape(-1)[indices] == values):
                    raise ValueError("unchanged entries must not be counted as edits")
            result.append((name, indices.copy(), values.copy()))
            receipts.append({"name": name, "base_sha256": spec["sha256"],
                             "update_sha256": update_hash, "changed_entries": len(indices)})
    return result, {"patch_sha256": sha(data), "changed_entries": count, "updates": receipts}


def pack(updates: list[tuple[dict[str, JSON], NDArray[np.int64], NDArray[np.uint16]]]) -> bytes:
    arrays: dict[str, NDArray[np.int64] | NDArray[np.uint16] | NDArray[np.uint8]] = {}
    entries: list[JSON] = []
    for n, (spec, indices, values) in enumerate(updates):
        entries.append({"name": spec["name"], "shape": spec["shape"], "dtype": spec["dtype"],
                        "base_sha256": spec["sha256"],
                        "update_sha256": sha(indices.tobytes() + values.tobytes())})
        arrays[f"i{n}"], arrays[f"v{n}"] = indices, values
    arrays["meta"] = np.frombuffer(canonical({"schema": "WSPPatch/1", "updates": entries}), dtype=np.uint8)
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()
