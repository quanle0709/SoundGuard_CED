import huggingface_hub

import sound_classifier


def test_frozen_local_snapshot_is_preferred(monkeypatch):
    calls = []

    def fake_snapshot_download(model_id, *, revision, local_files_only):
        calls.append((model_id, revision, local_files_only))
        return r"C:\frozen\ced-tiny"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    source, kwargs = sound_classifier._resolve_model_source()

    assert source == sound_classifier.MODEL_ID
    assert kwargs == {
        "revision": sound_classifier.MODEL_REVISION,
        "local_files_only": True,
    }
    assert calls == [(
        sound_classifier.MODEL_ID,
        sound_classifier.MODEL_REVISION,
        True,
    )]


def test_missing_local_snapshot_uses_same_pinned_revision(monkeypatch):
    def missing_snapshot(*_args, **_kwargs):
        raise OSError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing_snapshot)

    source, kwargs = sound_classifier._resolve_model_source()

    assert source == sound_classifier.MODEL_ID
    assert kwargs == {"revision": sound_classifier.MODEL_REVISION}


def test_current_cached_snapshot_contains_frozen_preprocessor():
    source, kwargs = sound_classifier._resolve_model_source()

    assert source == sound_classifier.MODEL_ID
    assert kwargs == {
        "revision": sound_classifier.MODEL_REVISION,
        "local_files_only": True,
    }
