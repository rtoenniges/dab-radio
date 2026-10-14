"""A paho-mqtt client that connects to nothing, for the tests of `net.mqtt`: it records
what the bridge sets up and publishes, and the test plays the broker by calling the
callbacks the bridge installed."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, ClassVar

from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.reasoncodes import ReasonCode


def reason(name: str, packet: int = PacketTypes.CONNACK) -> ReasonCode:
    """A reason code as paho hands it to the callbacks, e.g. "Success" or
    "Not authorized"."""
    return ReasonCode(packet, name)


class FakeMqttClient:
    instances: ClassVar[list[FakeMqttClient]] = []

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.username: str | None = None
        self.password: str | None = None
        self.tls: dict | None = None
        self.insecure = False
        self.will: tuple | None = None
        self.reconnect_delay: tuple | None = None
        self.connect_args: tuple | None = None
        self.loop_running = False
        self.disconnected = False
        self.published: list[tuple[str, Any, bool]] = []  # topic, payload, retain
        self.qos: dict[str, int] = {}  # the QoS of the last publication to each topic
        self.subscribed: list = []
        self.on_connect = None
        self.on_connect_fail = None
        self.on_disconnect = None
        self.on_message = None
        FakeMqttClient.instances.append(self)

    # --- What the bridge calls --------------------------------------------------------

    def username_pw_set(self, username: str, password: str | None = None) -> None:
        self.username = username
        self.password = password

    def tls_set(self, **kwargs: Any) -> None:
        self.tls = kwargs

    def tls_insecure_set(self, value: bool) -> None:
        self.insecure = value

    def will_set(self, topic: str, payload: Any = None, qos: int = 0, retain: bool = False) -> None:
        self.will = (topic, payload, retain)

    def reconnect_delay_set(self, min_delay: int, max_delay: int) -> None:
        self.reconnect_delay = (min_delay, max_delay)

    def connect_async(self, host: str, port: int = 1883, keepalive: int = 60) -> None:
        self.connect_args = (host, port, keepalive)

    def loop_start(self) -> None:
        self.loop_running = True

    def loop_stop(self) -> None:
        self.loop_running = False

    def disconnect(self) -> None:
        self.disconnected = True

    def publish(self, topic: str, payload: Any = None, qos: int = 0, retain: bool = False) -> None:
        self.published.append((topic, payload, retain))
        self.qos[topic] = qos

    def subscribe(self, topics: Any, qos: int = 0) -> None:
        self.subscribed.append(topics)

    # --- What the test does as the broker ---------------------------------------------

    def connect_now(self, name: str = "Success") -> None:
        self.on_connect(self, None, {}, reason(name), None)

    def drop(self, name: str = "Unspecified error") -> None:
        self.on_disconnect(self, None, {}, reason(name, PacketTypes.DISCONNECT), None)

    def fail(self) -> None:
        self.on_connect_fail(self, None)

    def deliver(self, topic: str, payload: bytes | str) -> None:
        if isinstance(payload, str):
            payload = payload.encode()
        self.on_message(self, None, SimpleNamespace(topic=topic, payload=payload))

    # --- What the test reads ----------------------------------------------------------

    def payloads(self, topic: str) -> list[Any]:
        return [payload for published, payload, _retain in self.published if published == topic]

    def last(self, topic: str) -> Any:
        payloads = self.payloads(topic)
        if not payloads:
            raise AssertionError(f"nothing published to {topic}")
        return payloads[-1]

    def last_json(self, topic: str) -> Any:
        return json.loads(self.last(topic))

    def topics(self) -> list[str]:
        return [topic for topic, _payload, _retain in self.published]
