"""Stored programme guides per channel: the objects of an SPI carousel as received.

One turn of a programme guide's carousel takes minutes (EPG Deutschland on 5C: 11
minutes). So that the guide is there right away the next time the channel comes up, it
is stored like the service list in `dab.ensemble_cache`: one directory per channel in
the application's data directory, with an index (JSON) and one file per MOT object
holding its body exactly as received. The SI and PI objects are decoded again on
loading (for a whole week of three services some 40 ms), so the stored format does not
depend on the decoded model. The service labels of the FIC are stored along with them:
a carousel may not have brought its SI (with the names) yet, and the channel's service
list need not be stored.

Unlike the service list, storing happens automatically: every object is complete in
itself (CRC-checked data groups, fully reassembled), so taking one over can never make
the stored guide worse, and a new MOT directory removes what the station no longer
broadcasts (`retain`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from dab.mot import MotObject

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1
_INDEX_FILENAME = "index.json"


@dataclass(frozen=True)
class StoredGuide:
    """What is stored for one channel."""

    objects: tuple[MotObject, ...]
    updated_at: float  # Unix time of the most recent change
    labels: dict[int, str]  # SId -> service label from the FIC


def _object_filename(content_name: str) -> str:
    """ContentNames may contain anything, file names may not."""
    return hashlib.sha1(content_name.encode("utf-8")).hexdigest()[:20] + ".bin"


def _body_digest(body: bytes) -> str:
    return hashlib.sha1(body).hexdigest()


class GuideCache:
    """The stored programme guides, one directory per channel below `directory`.

    The index of a channel is kept in memory once read; all methods run in the GUI
    thread.

    Raises:
        OSError: from `store`, `retain` and `remove` if the files cannot be written.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._indexes: dict[str, dict] = {}

    def channels(self) -> list[str]:
        """The channels a guide is stored for. An unreadable directory counts as none
        (with a warning)."""
        try:
            directories = [path for path in self._directory.iterdir() if path.is_dir()]
        except FileNotFoundError:
            return []
        except OSError as exc:
            logger.warning("Stored guides in %s are not readable: %s", self._directory, exc)
            return []
        return sorted(
            directory.name for directory in directories if (directory / _INDEX_FILENAME).is_file()
        )

    def load(self, channel: str) -> StoredGuide | None:
        """The stored guide of a channel, None if there is none. Unreadable entries are
        skipped with a warning; a corrupt index counts as no guide."""
        index = self._index(channel)
        if not index["objects"]:
            return None
        labels = {}
        for sid, label in index.get("labels", {}).items():
            try:
                labels[int(sid, 16)] = str(label)
            except TypeError, ValueError:
                logger.warning("Skipped stored label %r of %s", sid, channel)
        objects = []
        for name, entry in index["objects"].items():
            try:
                body = (self._channel_directory(channel) / entry["file"]).read_bytes()
                objects.append(
                    MotObject(
                        transport_id=0,
                        content_type=int(entry["content_type"]),
                        content_subtype=int(entry["content_subtype"]),
                        content_name=name,
                        body=body,
                        parameters={
                            int(key): bytes.fromhex(value)
                            for key, value in entry["parameters"].items()
                        },
                    )
                )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipped stored guide object %r of %s: %s", name, channel, exc)
        return StoredGuide(tuple(objects), float(index["updated_at"]), labels)

    def store(self, channel: str, mot_object: MotObject, now: float) -> bool:
        """Stores an object, replacing one with the same ContentName. An unchanged
        object (the same body again) is not written a second time.

        Returns:
            True if the object was written, False if it was stored already (or has no
            ContentName to store it by).
        """
        name = mot_object.content_name
        if not name:
            return False
        index = self._index(channel)
        digest = _body_digest(mot_object.body)
        entry = {
            "file": _object_filename(name),
            "content_type": mot_object.content_type,
            "content_subtype": mot_object.content_subtype,
            "parameters": {str(key): value.hex() for key, value in mot_object.parameters.items()},
            "sha1": digest,
        }
        if index["objects"].get(name) == entry:
            return False
        directory = self._channel_directory(channel)
        directory.mkdir(parents=True, exist_ok=True)
        _write_atomically(directory / entry["file"], mot_object.body)
        index["objects"][name] = entry
        index["updated_at"] = now
        self._write_index(channel)
        return True

    def store_labels(self, channel: str, labels: dict[int, str]) -> None:
        """Stores the service labels of the FIC, in addition to those stored before
        (the FIC may not have delivered all of them yet). Only once there is a guide."""
        index = self._index(channel)
        stored = index.setdefault("labels", {})
        new = {f"{sid:X}": label for sid, label in labels.items() if label}
        if not index["objects"] or all(stored.get(sid) == label for sid, label in new.items()):
            return
        stored.update(new)
        self._write_index(channel)

    def retain(self, channel: str, content_names: set[str], now: float) -> None:
        """Deletes every object whose ContentName is not in `content_names`."""
        index = self._index(channel)
        outdated = [name for name in index["objects"] if name not in content_names]
        if not outdated:
            return
        directory = self._channel_directory(channel)
        for name in outdated:
            entry = index["objects"].pop(name)
            (directory / entry["file"]).unlink(missing_ok=True)
        index["updated_at"] = now
        self._write_index(channel)

    def remove(self, channel: str) -> None:
        """Deletes the stored guide of a channel altogether."""
        self._indexes.pop(channel, None)
        directory = self._channel_directory(channel)
        if directory.exists():
            shutil.rmtree(directory)

    def _channel_directory(self, channel: str) -> Path:
        if not channel or any(part in channel for part in ("/", "\\", "..")):
            raise ValueError(f"Invalid channel name for the guide cache: {channel!r}")
        return self._directory / channel

    def _index(self, channel: str) -> dict:
        index = self._indexes.get(channel)
        if index is None:
            index = self._read_index(channel)
            self._indexes[channel] = index
        return index

    def _read_index(self, channel: str) -> dict:
        empty = {"version": _FORMAT_VERSION, "updated_at": 0.0, "labels": {}, "objects": {}}
        path = self._channel_directory(channel) / _INDEX_FILENAME
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return empty
        except (OSError, ValueError) as exc:
            logger.warning("Stored guide %s is not readable: %s", path, exc)
            return empty
        if (
            not isinstance(data, dict)
            or data.get("version") != _FORMAT_VERSION
            or not isinstance(data.get("objects"), dict)
        ):
            logger.warning("Stored guide %s has an unknown format", path)
            return empty
        return data

    def _write_index(self, channel: str) -> None:
        directory = self._channel_directory(channel)
        directory.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self._indexes[channel], ensure_ascii=False, indent=1)
        _write_atomically(directory / _INDEX_FILENAME, text.encode("utf-8"))


def _write_atomically(path: Path, data: bytes) -> None:
    """Writes to a temporary file first and renames it, so that an interruption cannot
    leave a half-written file behind (as `dab.ensemble_cache.EnsembleCache.save`)."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
