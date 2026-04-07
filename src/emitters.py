from __future__ import annotations
import json
from typing import Any, Dict, Iterable
from .common import ensure_parent
try:
    from kafka import KafkaProducer
except Exception:
    KafkaProducer = None

class EventEmitter:

    def emit(self, payload: Dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return

class JsonlEmitter(EventEmitter):

    def __init__(self, output_path: str):
        ensure_parent(output_path)
        self.output_path = output_path
        self._f = open(output_path, 'w', encoding='utf-8')

    def emit(self, payload: Dict[str, Any]) -> None:
        self._f.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def close(self) -> None:
        self._f.flush()
        self._f.close()

class KafkaJsonEmitter(EventEmitter):

    def __init__(self, bootstrap_servers: str, topic: str):
        if KafkaProducer is None:
            raise RuntimeError('kafka-python is not installed. Install with `pip install kafka-python`.')
        self.topic = topic
        self.producer = KafkaProducer(bootstrap_servers=[s.strip() for s in bootstrap_servers.split(',') if s.strip()], value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'), key_serializer=lambda k: k.encode('utf-8'), linger_ms=25)

    def emit(self, payload: Dict[str, Any]) -> None:
        self.producer.send(self.topic, key=payload['asset_id'].encode('utf-8'), value=payload)

    def close(self) -> None:
        self.producer.flush()
        self.producer.close()

class CompositeEmitter(EventEmitter):

    def __init__(self, emitters: Iterable[EventEmitter]):
        self.emitters = list(emitters)

    def emit(self, payload: Dict[str, Any]) -> None:
        for emitter in self.emitters:
            emitter.emit(payload)

    def close(self) -> None:
        for emitter in self.emitters:
            emitter.close()
