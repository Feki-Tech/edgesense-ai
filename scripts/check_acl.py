"""Prove per-device ACL isolation on the secured uplink broker.

Acceptance check for PLATFORM.md phase 1 (§4.2/§4.4): a device credential can
publish only under its own `es/<org>/<site>/<machine>/…` prefix and read only
its own control topic. Publish denials are verified two ways: the broker's
MQTT v5 PUBACK reason code (135 = not authorized; mosquitto acks allowed
publishes with 0 or 16 = "no matching subscribers"), and an independent ops
observer that must receive exactly the allowed marker publishes and nothing
else. Read ACLs are verified by delivery: mosquitto accepts any SUBSCRIBE but
silently drops messages the client may not read, so the check publishes
markers to the device's own and a foreign control topic and asserts only the
own-topic marker arrives.

Requires the secured stack: `make stack-secure` (uplink broker on :12883),
demo credentials from deploy/secure/passwd.

    python scripts/check_acl.py            (or: make check-acl)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid

import paho.mqtt.client as mqtt

HOST = os.environ.get("EDGESENSE_BROKER_HOST", "localhost")
PORT = int(os.environ.get("EDGESENSE_UPLINK_PORT", "12883"))

DEVICE = os.environ.get("EDGESENSE_CHECK_DEVICE", "default/default/machine-01")
DEVICE_PW = os.environ.get("EDGESENSE_CHECK_DEVICE_PW", "machine-01-demo-pw")
OPS = os.environ.get("EDGESENSE_CHECK_OPS", "ops")
OPS_PW = os.environ.get("EDGESENSE_CHECK_OPS_PW", "ops-demo-pw")
FOREIGN_PREFIX = os.environ.get("EDGESENSE_CHECK_FOREIGN", "es/acme/lyon/pump-07")

OWN_PREFIX = "es/" + DEVICE
SIBLING_PREFIX = OWN_PREFIX.rsplit("/", 1)[0] + "/machine-02"
NOT_AUTHORIZED = 135
# MQTT v5 PUBACK: 0x00 Success, 0x10 Success but no matching subscribers.
PUB_OK = (0, 16)
TIMEOUT = 5.0
# unique payload so concurrent real traffic (the live agent) can't pollute checks
NONCE = uuid.uuid4().hex
MARKER = json.dumps({"check": "acl", "nonce": NONCE}).encode()


class V5Client:
    """Thin synchronous wrapper exposing MQTT v5 reason codes."""

    def __init__(self, client_id: str, username: str | None, password: str) -> None:
        self.connected = threading.Event()
        self.connect_rc: int | None = None
        self.pub_rcs: dict[int, int] = {}
        self.sub_rcs: dict[int, int] = {}
        self.messages: list[tuple[str, bytes]] = []
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=client_id, protocol=mqtt.MQTTv5)
        if username:
            self.client.username_pw_set(username, password)
        self.client.on_connect = self._on_connect
        self.client.on_publish = self._on_publish
        self.client.on_subscribe = self._on_subscribe
        self.client.on_message = self._on_message

    def _on_connect(self, _c, _u, _flags, reason_code, _props=None) -> None:
        if self.connect_rc is None:
            self.connect_rc = reason_code.value
        self.connected.set()

    def _on_publish(self, _c, _u, mid, reason_code, _props=None) -> None:
        self.pub_rcs[mid] = reason_code.value

    def _on_subscribe(self, _c, _u, mid, reason_codes, _props=None) -> None:
        self.sub_rcs[mid] = reason_codes[0].value

    def _on_message(self, _c, _u, msg) -> None:
        self.messages.append((msg.topic, msg.payload))

    def connect(self) -> int | None:
        """Returns the CONNACK reason code (None if no CONNACK arrived)."""
        try:
            self.client.connect(HOST, PORT)
        except OSError as exc:
            print(f"cannot reach broker at {HOST}:{PORT} — is the secured stack up? "
                  f"(make stack-secure): {exc}")
            sys.exit(1)
        self.client.loop_start()
        self.connected.wait(TIMEOUT)
        return self.connect_rc

    def publish_rc(self, topic: str) -> int | None:
        """QoS-1 publish of the marker payload; returns the PUBACK reason code."""
        info = self.client.publish(topic, MARKER, qos=1)
        return self._await(self.pub_rcs, info.mid)

    def marker_on(self, topic: str, timeout: float = TIMEOUT) -> bool:
        """True if the marker payload arrives on `topic` within `timeout`."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(t == topic and p == MARKER for t, p in self.messages):
                return True
            time.sleep(0.05)
        return False

    def subscribe_rc(self, topic_filter: str) -> int | None:
        """Returns the SUBACK reason code (None on timeout)."""
        _, mid = self.client.subscribe(topic_filter, qos=1)
        return self._await(self.sub_rcs, mid)

    @staticmethod
    def _await(store: dict[int, int], mid: int) -> int | None:
        deadline = time.time() + TIMEOUT
        while time.time() < deadline:
            if mid in store:
                return store[mid]
            time.sleep(0.05)
        return None

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()


def main() -> int:
    results: list[tuple[bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((ok, name))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    print(f"=== EdgeSense ACL isolation check — {HOST}:{PORT}, device {DEVICE!r} ===\n")

    observer = V5Client("edgesense-acl-observer", OPS, OPS_PW)
    rc = observer.connect()
    if rc != 0:
        print(f"ops observer could not connect (rc={rc}) — wrong stack or credentials?")
        return 1
    observer.subscribe_rc("es/+/+/+/events")
    time.sleep(0.3)

    device = V5Client("edgesense-acl-device", DEVICE, DEVICE_PW)
    rc = device.connect()
    check("device credential connects", rc == 0, f"rc={rc}")

    print("\n-- publish --")
    rc = device.publish_rc(f"{OWN_PREFIX}/events")
    check("own events topic allowed", rc in PUB_OK, f"rc={rc}")
    rc = device.publish_rc(f"{OWN_PREFIX}/sensors/vibration")
    check("own sensors topic allowed", rc in PUB_OK, f"rc={rc}")
    rc = device.publish_rc(f"{FOREIGN_PREFIX}/events")
    check("foreign-org prefix denied", rc == NOT_AUTHORIZED, f"rc={rc}")
    rc = device.publish_rc(f"{SIBLING_PREFIX}/events")
    check("sibling-machine prefix denied", rc == NOT_AUTHORIZED, f"rc={rc}")

    print("\n-- subscribe (mosquitto grants SUBACKs and enforces read ACLs at"
          " delivery time; proven with ops-published markers) --")
    device.subscribe_rc(f"{OWN_PREFIX}/control")
    device.subscribe_rc(f"{FOREIGN_PREFIX}/control")
    time.sleep(0.3)
    observer.client.publish(f"{OWN_PREFIX}/control", MARKER, qos=1)
    observer.client.publish(f"{FOREIGN_PREFIX}/control", MARKER, qos=1)
    check("own control topic delivered",
          device.marker_on(f"{OWN_PREFIX}/control"))
    check("foreign control topic not delivered",
          not device.marker_on(f"{FOREIGN_PREFIX}/control", timeout=2.0))

    print("\n-- authentication --")
    bad = V5Client("edgesense-acl-badpw", DEVICE, "wrong-password")
    rc = bad.connect()
    check("wrong password rejected", rc != 0, f"rc={rc}")
    bad.stop()
    anon = V5Client("edgesense-acl-anon", None, "")
    rc = anon.connect()
    check("anonymous connect rejected", rc != 0, f"rc={rc}")
    anon.stop()

    print("\n-- observer cross-check --")
    time.sleep(1.0)  # let any stray delivery land
    seen = [t for t, p in observer.messages if p == MARKER]
    check("allowed event delivered", f"{OWN_PREFIX}/events" in seen, f"seen={seen}")
    leaked = [t for t in seen
              if t not in (f"{OWN_PREFIX}/events",
                           f"{OWN_PREFIX}/control", f"{FOREIGN_PREFIX}/control")]
    check("denied publishes not delivered", not leaked, f"leaked={leaked}")

    device.stop()
    observer.stop()

    failed = [name for ok, name in results if not ok]
    print(f"\nVERDICT: {'PASS' if not failed else 'FAIL'} "
          f"({len(results) - len(failed)}/{len(results)} checks)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
