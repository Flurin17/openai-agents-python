from __future__ import annotations

import logging

from agents.tracing import (
    add_trace_processor,
    processors as tracing_processors,
    set_auto_replace_trace_processor_on_add,
    setup as tracing_setup,
)
from agents.tracing.processor_interface import TracingProcessor
from agents.tracing.provider import DefaultTraceProvider
from agents.tracing.span_data import CustomSpanData


class DummyProcessor(TracingProcessor):
    def on_trace_start(self, trace) -> None:
        return None

    def on_trace_end(self, trace) -> None:
        return None

    def on_span_start(self, span) -> None:
        return None

    def on_span_end(self, span) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self) -> None:
        return None


class RecordingProcessor(TracingProcessor):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_trace_start(self, trace) -> None:
        self.events.append("trace_start")

    def on_trace_end(self, trace) -> None:
        self.events.append("trace_end")

    def on_span_start(self, span) -> None:
        self.events.append("span_start")

    def on_span_end(self, span) -> None:
        self.events.append("span_end")

    def shutdown(self) -> None:
        return None

    def force_flush(self) -> None:
        return None


def _reset_global_tracing_state(monkeypatch) -> None:
    monkeypatch.setattr(tracing_setup, "GLOBAL_TRACE_PROVIDER", None)
    monkeypatch.setattr(tracing_setup, "_SHUTDOWN_HANDLER_REGISTERED", False)
    monkeypatch.setattr(tracing_processors, "_global_exporter", None)
    monkeypatch.setattr(tracing_processors, "_global_processor", None)


def _registered_processors() -> tuple[TracingProcessor, ...]:
    provider = tracing_setup.get_trace_provider()
    assert isinstance(provider, DefaultTraceProvider)
    return provider._multi_processor.get_processors()


def test_cold_start_default_provider_registers_openai_batch_processor(monkeypatch) -> None:
    _reset_global_tracing_state(monkeypatch)

    processors = _registered_processors()

    assert len(processors) == 1
    assert isinstance(processors[0], tracing_processors.BatchTraceProcessor)


def test_add_trace_processor_keeps_default_and_warns_once(monkeypatch, caplog) -> None:
    _reset_global_tracing_state(monkeypatch)
    set_auto_replace_trace_processor_on_add(False)

    with caplog.at_level(logging.WARNING, logger="openai.agents"):
        add_trace_processor(DummyProcessor())
        add_trace_processor(DummyProcessor())

    processors = _registered_processors()
    assert len(processors) == 3
    assert isinstance(processors[0], tracing_processors.BatchTraceProcessor)
    assert isinstance(processors[1], DummyProcessor)
    assert isinstance(processors[2], DummyProcessor)

    warnings = [
        record.message
        for record in caplog.records
        if "set_auto_replace_trace_processor_on_add(True)" in record.message
    ]
    assert len(warnings) == 1


def test_add_trace_processor_replaces_default_when_opted_in(monkeypatch, caplog) -> None:
    _reset_global_tracing_state(monkeypatch)
    set_auto_replace_trace_processor_on_add(True)

    with caplog.at_level(logging.WARNING, logger="openai.agents"):
        add_trace_processor(DummyProcessor())

    processors = _registered_processors()
    assert len(processors) == 1
    assert isinstance(processors[0], DummyProcessor)
    assert caplog.records == []


def test_opted_in_repeated_add_does_not_reintroduce_default(monkeypatch) -> None:
    _reset_global_tracing_state(monkeypatch)
    set_auto_replace_trace_processor_on_add(True)

    first = DummyProcessor()
    second = DummyProcessor()
    add_trace_processor(first)
    add_trace_processor(second)

    processors = _registered_processors()
    assert len(processors) == 2
    assert processors == (first, second)


def test_disable_tracing_disables_events_even_with_custom_processors(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_AGENTS_DISABLE_TRACING", "1")

    provider = DefaultTraceProvider()
    recorder = RecordingProcessor()
    provider.set_processors([recorder])

    trace = provider.create_trace("disabled")
    trace.start()
    span = provider.create_span(CustomSpanData(name="test_span", data={}), parent=trace)
    span.start()
    span.finish()
    trace.finish()

    assert recorder.events == []
