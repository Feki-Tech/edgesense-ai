"""Model artifact signing and verification (Ed25519).

A trained bundle is a joblib pickle: loading one is arbitrary code execution,
so *whoever can write the artifact owns the inference container*
(``docs/SECURITY.md`` S6/P4). Once a model can arrive over the air — from the
MLflow registry (``inference/registry.py``) rather than baked into the image —
"the file is on disk" stops being evidence that it came from the training
pipeline. Signing closes that gap: training signs the artifact bytes, serving
verifies them **before** ``joblib.load`` ever sees them.

Layout — a detached sidecar next to the bundle::

    ml/model/model.joblib          the artifact
    ml/model/model.joblib.sig      this envelope (JSON)

The signed message is a domain-separated digest of the artifact bytes::

    b"edgesense-model-signature-v1\\n" + sha256_hex(artifact)

The domain prefix keeps a signature from being replayed as one over some other
kind of EdgeSense payload. Signing the digest rather than the file itself keeps
the operation constant-time in artifact size and lets a verifier report
"digest mismatch" (corruption/truncation) separately from "bad signature"
(wrong or untrusted key).

The bundle's manifest is *inside* the artifact, so the digest covers it too;
the sidecar ``model.manifest.json`` is a derivative convenience copy and is
deliberately not signed on its own.

Configuration (all optional — unsigned bundles keep working by default):

``EDGESENSE_SIGNING_KEY``
    Private key for signing: a filesystem path, or inline PEM text. When unset,
    ``save_bundle`` writes an unsigned artifact exactly as before.
``EDGESENSE_TRUSTED_KEYS``
    Public keys a verifier accepts: a path, or inline PEM text. Several PEM
    blocks may be concatenated, which is what makes key rotation possible —
    trust the new key alongside the old one, re-sign, then drop the old.
``EDGESENSE_REQUIRE_SIGNATURE``
    When truthy, an artifact that cannot be *positively* verified is refused.
    Off by default so existing unsigned bundles (and every image built before
    this feature) keep serving; turn it on once your delivery path signs.

    python ml/signing.py keygen --out ml/keys/edgesense.pem
    python ml/signing.py sign ml/model/model.joblib
    python ml/signing.py verify ml/model/model.joblib
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

DOMAIN = b"edgesense-model-signature-v1\n"
SIGNATURE_SCHEMA_VERSION = 1
ALGORITHM = "ed25519"
SIGNATURE_SUFFIX = ".sig"

# verification outcomes (also the label values of the Prometheus counter)
VERIFIED = "verified"      # signature present, trusted key, digest matches
UNSIGNED = "unsigned"      # no signature sidecar at all
UNVERIFIABLE = "unverifiable"  # signature present but no trust store configured


class SigningError(RuntimeError):
    """The artifact could not be signed (missing/invalid key, no library)."""


class VerificationError(RuntimeError):
    """The artifact must not be loaded: tampered, untrusted, or required-but-unsigned."""


def _crypto():
    """Import the Ed25519 primitives lazily with an actionable error."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey, Ed25519PublicKey)
    except ImportError as exc:  # pragma: no cover - cryptography is a base dep
        raise SigningError(
            "the 'cryptography' package is required for model signing"
        ) from exc
    return InvalidSignature, serialization, Ed25519PrivateKey, Ed25519PublicKey


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _pem_bytes(source: str) -> bytes:
    """Read PEM material given either inline PEM text or a path to it."""
    if "-----BEGIN" in source:
        return source.encode()
    path = Path(source.strip())
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SigningError(f"could not read key material from {path}: {exc}") from exc


def signature_path(artifact: "Path | str") -> Path:
    """Sidecar path for an artifact: model.joblib -> model.joblib.sig."""
    p = Path(artifact)
    return p.with_name(p.name + SIGNATURE_SUFFIX)


def file_sha256(path: "Path | str") -> str:
    """Streaming sha256 of a file's bytes (hex)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed_message(sha256_hex: str) -> bytes:
    """The exact byte string Ed25519 signs for a given artifact digest."""
    return DOMAIN + sha256_hex.encode()


def key_id(public_key) -> str:
    """Short stable identifier of a public key (sha256 of its raw bytes)."""
    _, serialization, _, _ = _crypto()
    raw = public_key.public_bytes(serialization.Encoding.Raw,
                                  serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def generate_keypair() -> tuple[bytes, bytes]:
    """Fresh Ed25519 keypair as (private PEM, public PEM)."""
    _, serialization, Ed25519PrivateKey, _ = _crypto()
    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption())
    public_pem = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo)
    return private_pem, public_pem


def load_private_key(source: str | None = None):
    """Load the signing key from ``source`` or ``EDGESENSE_SIGNING_KEY``.

    Returns None when neither is configured — callers treat that as "signing
    is not enabled here" rather than an error, so unsigned workflows (local
    ``make train``, CI, existing images) keep working untouched.
    """
    _, serialization, Ed25519PrivateKey, _ = _crypto()
    source = source or os.environ.get("EDGESENSE_SIGNING_KEY")
    if not source:
        return None
    try:
        key = serialization.load_pem_private_key(_pem_bytes(source), password=None)
    except SigningError:
        raise
    except Exception as exc:
        raise SigningError(f"could not parse the signing key: {exc}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise SigningError(f"signing key is not Ed25519 (got {type(key).__name__})")
    return key


def load_trusted_keys(source: str | None = None) -> dict[str, object]:
    """Trusted public keys by key id, from ``source`` or ``EDGESENSE_TRUSTED_KEYS``.

    Concatenated PEM blocks are all accepted, so a rotation can trust the
    outgoing and incoming key at the same time.
    """
    _, serialization, _, Ed25519PublicKey = _crypto()
    source = source or os.environ.get("EDGESENSE_TRUSTED_KEYS")
    if not source:
        return {}

    pem = _pem_bytes(source)
    marker = b"-----END PUBLIC KEY-----"
    blocks = [block + marker for block in pem.split(marker) if b"-----BEGIN" in block]
    if not blocks:
        raise SigningError("trusted key material contains no PUBLIC KEY blocks")

    trusted: dict[str, object] = {}
    for block in blocks:
        try:
            key = serialization.load_pem_public_key(block)
        except Exception as exc:
            raise SigningError(f"could not parse a trusted public key: {exc}") from exc
        if not isinstance(key, Ed25519PublicKey):
            raise SigningError(
                f"trusted key is not Ed25519 (got {type(key).__name__})")
        trusted[key_id(key)] = key
    return trusted


def build_envelope(artifact: "Path | str", private_key, *,
                   model_version: str | None = None,
                   artifact_name: str | None = None) -> dict:
    """Sign an artifact's digest and return the signature envelope.

    ``artifact_name`` records a name other than the file's own — needed when
    signing a temp file that is about to be renamed into place.
    """
    sha256_hex = file_sha256(artifact)
    signature = private_key.sign(signed_message(sha256_hex))
    return {
        "schema_version": SIGNATURE_SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "artifact": artifact_name or Path(artifact).name,
        "artifact_sha256": sha256_hex,
        "key_id": key_id(private_key.public_key()),
        "signature": base64.b64encode(signature).decode(),
        "signed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_version": model_version,
    }


def write_envelope(envelope: dict, path: "Path | str") -> Path:
    """Atomically write a signature sidecar (temp file + os.replace)."""
    path = Path(path)
    data = json.dumps(envelope, indent=2).encode()
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def sign_artifact(artifact: "Path | str", private_key=None, *,
                  model_version: str | None = None,
                  sig_path: "Path | str | None" = None,
                  artifact_name: str | None = None) -> dict | None:
    """Sign ``artifact`` and write its sidecar. Returns None when no key is set.

    ``sig_path``/``artifact_name`` let a caller sign a temp file while naming
    the sidecar and the envelope after the final artifact — that is how
    ``ml/manifest.save_bundle`` publishes the signature before the bundle it
    describes lands on disk.
    """
    private_key = private_key or load_private_key()
    if private_key is None:
        return None
    envelope = build_envelope(artifact, private_key, model_version=model_version,
                              artifact_name=artifact_name)
    write_envelope(envelope, sig_path or signature_path(artifact))
    return envelope


def verify_artifact(artifact: "Path | str", *, trusted: dict | None = None,
                    require: bool | None = None) -> tuple[str, dict | None]:
    """Verify an artifact before it is loaded. Returns (status, envelope).

    Status is ``verified``, ``unsigned`` (no sidecar) or ``unverifiable``
    (signed, but this node has no trust store to check it against). Raises
    ``VerificationError`` when the artifact must not be loaded:

    - the sidecar is malformed, or names an algorithm we do not implement,
    - the digest does not match the bytes on disk (corrupt or swapped file),
    - the signature does not verify, or was made by an untrusted key,
    - ``require`` is on and the artifact is not positively verified.

    A *present but invalid* signature is always fatal, even when signatures are
    not required: that is tamper evidence, not a legacy artifact.
    """
    artifact = Path(artifact)
    require = _truthy(os.environ.get("EDGESENSE_REQUIRE_SIGNATURE")) \
        if require is None else require
    trusted = load_trusted_keys() if trusted is None else trusted
    sig_file = signature_path(artifact)

    if not sig_file.exists():
        if require:
            raise VerificationError(
                f"{artifact.name} is unsigned and EDGESENSE_REQUIRE_SIGNATURE is set "
                f"(expected {sig_file.name})")
        return UNSIGNED, None

    try:
        envelope = json.loads(sig_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerificationError(f"unreadable signature {sig_file.name}: {exc}") from exc
    if not isinstance(envelope, dict):
        raise VerificationError(f"signature {sig_file.name} is not an object")

    algorithm = envelope.get("algorithm")
    if algorithm != ALGORITHM:
        raise VerificationError(
            f"unsupported signature algorithm {algorithm!r} (expected {ALGORITHM!r})")

    expected = envelope.get("artifact_sha256")
    actual = file_sha256(artifact)
    if expected != actual:
        raise VerificationError(
            f"{artifact.name} does not match its signature: sha256 {actual[:16]}… "
            f"but the signature covers {str(expected)[:16]}…")

    if not trusted:
        if require:
            raise VerificationError(
                f"{artifact.name} is signed by key {envelope.get('key_id')} but no "
                f"trusted keys are configured (set EDGESENSE_TRUSTED_KEYS)")
        return UNVERIFIABLE, envelope

    signer = envelope.get("key_id")
    public_key = trusted.get(signer)
    if public_key is None:
        raise VerificationError(
            f"{artifact.name} is signed by untrusted key {signer!r} "
            f"(trusted: {', '.join(sorted(trusted)) or 'none'})")

    InvalidSignature, _, _, _ = _crypto()
    try:
        signature = base64.b64decode(envelope.get("signature", ""), validate=True)
    except Exception as exc:
        raise VerificationError(f"malformed signature encoding: {exc}") from exc
    try:
        public_key.verify(signature, signed_message(actual))
    except InvalidSignature as exc:
        raise VerificationError(
            f"signature on {artifact.name} does not verify against key {signer}") from exc
    return VERIFIED, envelope


def _cmd_keygen(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    public_out = out.with_name(out.stem + ".pub.pem")
    if out.exists() and not args.force:
        print(f"refusing to overwrite {out} (pass --force)", file=sys.stderr)
        return 1

    private_pem, public_pem = generate_keypair()
    # 0600 before any bytes land: a private key must never exist world-readable.
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(private_pem)
    public_out.write_bytes(public_pem)

    _, serialization, _, _ = _crypto()
    public = serialization.load_pem_public_key(public_pem)
    print(f"private key -> {out}  (keep secret, never commit)")
    print(f"public key  -> {public_out}")
    print(f"key id: {key_id(public)}")
    print(f"\nsign:   EDGESENSE_SIGNING_KEY={out} make train")
    print(f"verify: EDGESENSE_TRUSTED_KEYS={public_out} "
          f"EDGESENSE_REQUIRE_SIGNATURE=1 make inference")
    return 0


def _cmd_sign(args: argparse.Namespace) -> int:
    key = load_private_key(args.key)
    if key is None:
        print("no signing key: pass --key or set EDGESENSE_SIGNING_KEY",
              file=sys.stderr)
        return 1
    envelope = sign_artifact(args.artifact, key)
    print(f"signed {args.artifact} with key {envelope['key_id']} "
          f"-> {signature_path(args.artifact)}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    trusted = load_trusted_keys(args.trusted) if args.trusted else None
    try:
        status, envelope = verify_artifact(args.artifact, trusted=trusted,
                                           require=args.require)
    except VerificationError as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 1
    if status == VERIFIED:
        print(f"{status}: {args.artifact} signed by key {envelope['key_id']} "
              f"at {envelope['signed_at']}")
    elif status == UNVERIFIABLE:
        print(f"{status}: signature present (key {envelope['key_id']}) but no "
              f"trusted keys configured — set EDGESENSE_TRUSTED_KEYS")
    else:
        print(f"{status}: no {SIGNATURE_SUFFIX} sidecar next to {args.artifact}")
    return 0


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    kg = sub.add_parser("keygen", help="generate an Ed25519 signing keypair")
    kg.add_argument("--out", default="ml/keys/edgesense-signing.pem",
                    help="private key path (public key written alongside)")
    kg.add_argument("--force", action="store_true", help="overwrite an existing key")
    kg.set_defaults(func=_cmd_keygen)

    sg = sub.add_parser("sign", help="sign a model artifact")
    sg.add_argument("artifact")
    sg.add_argument("--key", default=None, help="private key (default: $EDGESENSE_SIGNING_KEY)")
    sg.set_defaults(func=_cmd_sign)

    vf = sub.add_parser("verify", help="verify a model artifact")
    vf.add_argument("artifact")
    vf.add_argument("--trusted", default=None,
                    help="trusted public keys (default: $EDGESENSE_TRUSTED_KEYS)")
    vf.add_argument("--require", action="store_true",
                    help="fail on an unsigned or unverifiable artifact")
    vf.set_defaults(func=_cmd_verify)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except SigningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
