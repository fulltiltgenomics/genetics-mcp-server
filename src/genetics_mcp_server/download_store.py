"""Disk-persisted download storage for tool result TSV files."""

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

EXPIRED_MESSAGE = "This download has expired. Please re-run the query to generate a new download link."


@dataclass
class DownloadMetadata:
    filename: str
    content_type: str
    created_at: float


class DownloadStore:
    """Stores download files on disk with JSON metadata sidecars."""

    def __init__(self, storage_path: str, ttl_seconds: int = 2592000):
        self._storage_path = storage_path
        self._ttl_seconds = ttl_seconds
        os.makedirs(storage_path, exist_ok=True)

    def store(self, data: bytes, filename: str, content_type: str = "text/tab-separated-values") -> str:
        """Store download data and return a unique ID."""
        download_id = uuid.uuid4().hex
        data_path = os.path.join(self._storage_path, f"{download_id}.tsv")
        meta_path = os.path.join(self._storage_path, f"{download_id}.json")

        meta = DownloadMetadata(
            filename=filename,
            content_type=content_type,
            created_at=time.time(),
        )

        with open(data_path, "wb") as f:
            f.write(data)
        with open(meta_path, "w") as f:
            json.dump(asdict(meta), f)

        logger.info(f"Stored download {download_id}: {filename} ({len(data)} bytes)")
        return download_id

    def get(self, download_id: str) -> tuple[bytes, str, str] | None:
        """Retrieve download by ID. Returns (data, filename, content_type) or None."""
        # validate ID to prevent path traversal
        if not download_id.isalnum():
            logger.error(f"Download request with invalid ID: {download_id}")
            return None

        data_path = os.path.join(self._storage_path, f"{download_id}.tsv")
        meta_path = os.path.join(self._storage_path, f"{download_id}.json")

        if not os.path.exists(data_path) or not os.path.exists(meta_path):
            logger.error(f"Download not found: {download_id}")
            return None

        try:
            with open(meta_path) as f:
                meta = DownloadMetadata(**json.load(f))
        except (json.JSONDecodeError, TypeError, KeyError):
            logger.error(f"Corrupt metadata for download {download_id}")
            return None

        elapsed = time.time() - meta.created_at
        if elapsed > self._ttl_seconds:
            logger.error(f"Download expired: {download_id} (age {elapsed:.0f}s, ttl {self._ttl_seconds}s)")
            self._remove(download_id)
            return None

        with open(data_path, "rb") as f:
            data = f.read()

        return data, meta.filename, meta.content_type

    def cleanup_expired(self) -> int:
        """Remove expired downloads. Returns number of entries removed."""
        removed = 0
        now = time.time()
        try:
            for entry in os.listdir(self._storage_path):
                if not entry.endswith(".json"):
                    continue
                meta_path = os.path.join(self._storage_path, entry)
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    if now - meta.get("created_at", 0) > self._ttl_seconds:
                        download_id = entry.removesuffix(".json")
                        self._remove(download_id)
                        removed += 1
                except (json.JSONDecodeError, OSError):
                    pass
        except OSError as e:
            logger.error(f"Error during download cleanup: {e}")
        if removed:
            logger.info(f"Cleaned up {removed} expired downloads")
        return removed

    def _remove(self, download_id: str) -> None:
        """Remove download files from disk."""
        for ext in (".tsv", ".json"):
            path = os.path.join(self._storage_path, f"{download_id}{ext}")
            try:
                os.remove(path)
            except OSError:
                pass


# singleton
_store: DownloadStore | None = None


def get_download_store() -> DownloadStore:
    """Get or create the singleton download store."""
    global _store
    if _store is None:
        from genetics_mcp_server.config import get_settings
        settings = get_settings()
        _store = DownloadStore(
            storage_path=settings.download_storage_path,
            ttl_seconds=settings.download_ttl_seconds,
        )
    return _store
