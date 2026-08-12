"""Model artifact signing: sign-on-save, verify-before-load, tamper refusal."""

from __future__ import annotations

import importlib
import json
import shutil

import joblib
import pytest
from fastapi.testclient import TestClient

from ml import signing
from ml.manifest import save_bundle


@pytest.fixture()
def keypair(tmp_path):
    """A signing keypair on disk as (private path, public path, key id)."""
    private_pem, public_pem = signing.generate_keypair()
    private = tmp_path / "signing.pem"
    public = tmp_path / "signing.pub.pem"
    private.write_bytes(private_pem)
    public.write_bytes(public_pem)
    key = signing.load_private_key(str(private))
    return private, public, signing.key_id(key.public_key())


@pytest.fixture()
def signed_bundle(small_model_path, keypair, tmp_path, monkeypatch):
    """A bundle written through save_bundle with signing enabled."""
    private, public, _ = keypair
    monkeypatch.setenv("EDGESENSE_SIGNING_KEY", str(private))
    monkeypatch.setenv("EDGESENSE_TRUSTED_KEYS", str(public))
    path = tmp_path / "signed" / "model.joblib"
    save_bundle(joblib.load(small_model_path), path)
    return path


# --- signing.py --------------------------------------------------------------

def test_sign_then_verify_roundtrip(signed_bundle, keypair) -> None:
    _, _, expected_key_id = keypair
    status, envelope = signing.verify_artifact(signed_bundle)
    assert status == signing.VERIFIED
    assert envelope["key_id"] == expected_key_id
    assert envelope["algorithm"] == "ed25519"
    assert envelope["artifact"] == "model.joblib"


def test_save_bundle_writes_a_signature_sidecar(signed_bundle) -> None:
    sidecar = signing.signature_path(signed_bundle)
    assert sidecar.exists() and sidecar.name == "model.joblib.sig"
    envelope = json.loads(sidecar.read_text())
    # the envelope records the version of the model it covers, so a signature
    # found next to an artifact is self-describing
    assert envelope["model_version"] == joblib.load(signed_bundle)["manifest"][
        "model_version"]


def test_tampered_artifact_is_refused(signed_bundle) -> None:
    data = bytearray(signed_bundle.read_bytes())
    data[len(data) // 2] ^= 0xFF
    signed_bundle.write_bytes(bytes(data))

    with pytest.raises(signing.VerificationError, match="does not match its signature"):
        signing.verify_artifact(signed_bundle)


def test_swapped_artifact_is_refused(signed_bundle, legacy_model_path) -> None:
    """A different, perfectly valid bundle under a signature is still a swap.

    This is the attack signing exists for: the substitute loads fine and scores
    fine, so every structural check in inference/server.py passes it.
    """
    shutil.copy(legacy_model_path, signed_bundle)
    with pytest.raises(signing.VerificationError, match="does not match its signature"):
        signing.verify_artifact(signed_bundle)


def test_untrusted_signer_is_refused(signed_bundle, tmp_path) -> None:
    _, other_public = signing.generate_keypair()
    other = tmp_path / "other.pub.pem"
    other.write_bytes(other_public)

    with pytest.raises(signing.VerificationError, match="untrusted key"):
        signing.verify_artifact(signed_bundle,
                                trusted=signing.load_trusted_keys(str(other)))


def test_forged_signature_is_refused(signed_bundle, keypair) -> None:
    """Right digest, wrong signer, envelope relabelled with the trusted key id."""
    _, public, trusted_key_id = keypair
    attacker_private, _ = signing.generate_keypair()
    attacker = signing.load_private_key(attacker_private.decode())

    envelope = signing.build_envelope(signed_bundle, attacker)
    envelope["key_id"] = trusted_key_id  # claim to be the key the node trusts
    signing.write_envelope(envelope, signing.signature_path(signed_bundle))

    with pytest.raises(signing.VerificationError, match="does not verify"):
        signing.verify_artifact(signed_bundle,
                                trusted=signing.load_trusted_keys(str(public)))


def test_unsigned_is_allowed_by_default_but_refused_when_required(
        small_model_path) -> None:
    assert signing.verify_artifact(small_model_path)[0] == signing.UNSIGNED
    with pytest.raises(signing.VerificationError, match="unsigned"):
        signing.verify_artifact(small_model_path, require=True)


def test_signed_without_trust_store_is_unverifiable(signed_bundle, monkeypatch) -> None:
    monkeypatch.delenv("EDGESENSE_TRUSTED_KEYS")
    assert signing.verify_artifact(signed_bundle)[0] == signing.UNVERIFIABLE
    # ...and that is not good enough when signatures are required
    with pytest.raises(signing.VerificationError, match="no trusted keys"):
        signing.verify_artifact(signed_bundle, require=True)


def test_require_signature_reads_the_environment(small_model_path, monkeypatch) -> None:
    monkeypatch.setenv("EDGESENSE_REQUIRE_SIGNATURE", "true")
    with pytest.raises(signing.VerificationError):
        signing.verify_artifact(small_model_path)


def test_trust_store_accepts_several_keys_for_rotation(signed_bundle, keypair,
                                                       tmp_path) -> None:
    _, public, _ = keypair
    _, incoming_public = signing.generate_keypair()
    rotation = tmp_path / "trusted.pem"
    rotation.write_bytes(incoming_public + public.read_bytes())

    trusted = signing.load_trusted_keys(str(rotation))
    assert len(trusted) == 2
    assert signing.verify_artifact(signed_bundle, trusted=trusted)[0] == signing.VERIFIED


def test_inline_pem_is_accepted_as_well_as_a_path(signed_bundle, keypair) -> None:
    """Container platforms inject config as env values, not files."""
    _, public, _ = keypair
    trusted = signing.load_trusted_keys(public.read_text())
    assert signing.verify_artifact(signed_bundle, trusted=trusted)[0] == signing.VERIFIED


def test_unsigned_save_removes_a_stale_signature(signed_bundle, small_model_path,
                                                 monkeypatch) -> None:
    """A sidecar describing a *different* artifact is worse than none."""
    monkeypatch.delenv("EDGESENSE_SIGNING_KEY")
    save_bundle(joblib.load(small_model_path), signed_bundle)

    assert not signing.signature_path(signed_bundle).exists()
    assert signing.verify_artifact(signed_bundle)[0] == signing.UNSIGNED


# --- serving path ------------------------------------------------------------

@pytest.fixture()
def served_signed(signed_bundle, monkeypatch):
    """Sidecar serving a signed bundle with the matching trust store."""
    monkeypatch.setenv("EDGESENSE_MODEL", str(signed_bundle))
    import inference.server as server
    importlib.reload(server)
    return TestClient(server.app), signed_bundle


def test_healthz_reports_signature_status(served_signed) -> None:
    client, _ = served_signed
    assert client.get("/healthz").json()["signature"] == signing.VERIFIED


def test_healthz_reports_unsigned_model(small_model_path, monkeypatch) -> None:
    monkeypatch.setenv("EDGESENSE_MODEL", str(small_model_path))
    monkeypatch.delenv("EDGESENSE_TRUSTED_KEYS", raising=False)
    import inference.server as server
    importlib.reload(server)
    assert TestClient(server.app).get("/healthz").json()["signature"] == signing.UNSIGNED


def test_reload_refuses_a_tampered_bundle_and_keeps_serving(served_signed) -> None:
    client, live_path = served_signed
    old_version = client.get("/healthz").json()["model_version"]

    data = bytearray(live_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    live_path.write_bytes(bytes(data))

    resp = client.post("/reload")
    assert resp.status_code == 400
    assert "does not match its signature" in resp.json()["detail"]

    # the champion is untouched and still scoring
    assert client.get("/healthz").json()["model_version"] == old_version
    assert client.post("/score", json={"vibration": 0.8, "temperature": 45.0,
                                       "current": 12.0}).status_code == 200


def test_reload_refuses_an_unsigned_replacement_when_required(
        served_signed, small_model_path, monkeypatch) -> None:
    client, live_path = served_signed
    monkeypatch.setenv("EDGESENSE_REQUIRE_SIGNATURE", "1")
    signing.signature_path(live_path).unlink()
    shutil.copy(small_model_path, live_path)

    resp = client.post("/reload")
    assert resp.status_code == 400
    assert "unsigned" in resp.json()["detail"]


def test_signature_checks_are_counted(served_signed) -> None:
    client, _ = served_signed
    body = client.get("/metrics").text
    assert 'edgesense_model_signature_checks_total{result="verified"}' in body


def test_verification_happens_before_the_pickle_is_loaded(served_signed,
                                                          monkeypatch) -> None:
    """The whole point: a rejected artifact must never reach joblib.load."""
    import inference.server as server

    client, live_path = served_signed
    data = bytearray(live_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    live_path.write_bytes(bytes(data))

    def explode(*_args, **_kwargs):
        raise AssertionError("joblib.load ran on an unverified artifact")

    monkeypatch.setattr(server.joblib, "load", explode)
    assert client.post("/reload").status_code == 400
