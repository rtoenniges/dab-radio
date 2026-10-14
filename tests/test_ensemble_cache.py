import json

from dab.ensemble_cache import (
    SOURCE_LIVE,
    SOURCE_SCAN,
    EnsembleCache,
    EnsembleInfo,
    keep_service_ensembles,
)


def _info(label: str | None = "Test Mux", **kwargs) -> EnsembleInfo:
    defaults = {
        "services": ((0xD001, "Radio Eins"), (0xD002, "Räterätsel")),
        "updated_at": 1_789_000_000.5,
        "source": SOURCE_SCAN,
        "fib_success": 0.97,
        "programme_types": ((0xD001, 10), (0xD002, 3)),
        "eid": 0x10BC,
        "frequencies": ((0x10BC, (178_352,)), (0x1205, (222_064, 227_360))),
        "service_ensembles": ((0xD001, (0x10BC, 0x1188)),),
    }
    return EnsembleInfo(label=label, **{**defaults, **kwargs})


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "sub" / "ensembles.json"
    cache = EnsembleCache(path)
    cache.put("5C", _info())
    cache.put("11D", _info(label=None, source=SOURCE_LIVE, fib_success=None))
    cache.save()

    reloaded = EnsembleCache(path)
    reloaded.load()

    assert reloaded.get("5C") == _info()
    assert reloaded.get("11D") == _info(label=None, source=SOURCE_LIVE, fib_success=None)
    assert reloaded.get("7B") is None


def test_entries_lists_all_channels(tmp_path):
    cache = EnsembleCache(tmp_path / "ensembles.json")
    cache.put("5C", _info())
    cache.put("7A", _info(label="NDR"))

    entries = cache.entries()
    entries.pop("5C")  # a copy: the cache keeps its entries

    assert entries == {"7A": _info(label="NDR")}
    assert set(cache.entries()) == {"5C", "7A"}


def test_remove_drops_channel(tmp_path):
    cache = EnsembleCache(tmp_path / "ensembles.json")
    cache.put("5C", _info())
    cache.remove("5C")
    cache.remove("5C")  # removing twice is not an error

    assert cache.get("5C") is None


def test_load_without_file_leaves_cache_empty(tmp_path):
    cache = EnsembleCache(tmp_path / "missing.json")
    cache.load()

    assert cache.get("5C") is None


def test_load_ignores_corrupt_file(tmp_path):
    path = tmp_path / "ensembles.json"
    path.write_text("{not json", encoding="utf-8")
    cache = EnsembleCache(path)
    cache.load()

    assert cache.get("5C") is None


def test_load_skips_invalid_entries_but_keeps_valid_ones(tmp_path):
    path = tmp_path / "ensembles.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "channels": {"5C": _info().to_json(), "7B": {"label": "no service list"}},
            }
        ),
        encoding="utf-8",
    )
    cache = EnsembleCache(path)
    cache.load()

    assert cache.get("5C") == _info()
    assert cache.get("7B") is None


def test_lists_stored_before_programme_types_load_without_them(tmp_path):
    """Programme types were added to the format later; older files stay valid."""
    path = tmp_path / "ensembles.json"
    entry = _info().to_json()
    del entry["programme_types"]
    path.write_text(json.dumps({"version": 1, "channels": {"5C": entry}}), encoding="utf-8")
    cache = EnsembleCache(path)
    cache.load()

    assert cache.get("5C") == _info(programme_types=())


def test_lists_stored_before_eid_and_frequencies_load_without_them(tmp_path):
    """EId, frequency information and the ensembles of the services (FIG 0/24) were
    added to the format later as well."""
    path = tmp_path / "ensembles.json"
    entry = _info().to_json()
    del entry["eid"]
    del entry["frequencies"]
    del entry["service_ensembles"]
    path.write_text(json.dumps({"version": 1, "channels": {"5C": entry}}), encoding="utf-8")
    cache = EnsembleCache(path)
    cache.load()

    assert cache.get("5C") == _info(eid=None, frequencies=(), service_ensembles=())


def test_a_new_entry_of_the_same_ensemble_keeps_the_fig_0_24_entries_it_lacks():
    """D001 comes anew and replaces the stored entry, D002 is kept."""
    previous = _info(service_ensembles=((0xD001, (0x10BC,)), (0xD002, (0x10BC, 0x1205))))
    info = _info(updated_at=1_789_000_100.0, service_ensembles=((0xD001, (0x10BC, 0x1188)),))

    kept = keep_service_ensembles(info, previous)

    assert kept == _info(
        updated_at=1_789_000_100.0,
        service_ensembles=((0xD001, (0x10BC, 0x1188)), (0xD002, (0x10BC, 0x1205))),
    )


def test_an_entry_of_another_ensemble_keeps_nothing():
    previous = _info(service_ensembles=((0xD002, (0x10BC,)),))

    assert keep_service_ensembles(_info(eid=0x10BD), previous) == _info(eid=0x10BD)
    assert keep_service_ensembles(_info(eid=None), previous) == _info(eid=None)
    assert keep_service_ensembles(_info(), None) == _info()
