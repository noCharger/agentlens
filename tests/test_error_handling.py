"""Tests for error classification and fault tolerance."""

from agentlens.eval.runner import (
    classify_error,
    ErrorKind,
    _extract_retry_delay,
    QuotaExhaustedError,
)


def test_classify_quota_exhausted():
    err = Exception("RESOURCE_EXHAUSTED 429 quotaValue 20")
    assert classify_error(err) == ErrorKind.QUOTA_EXHAUSTED


def test_classify_quota_with_retry_delay():
    err = Exception("429 RESOURCE_EXHAUSTED retryDelay: '40s'")
    assert classify_error(err) == ErrorKind.QUOTA_EXHAUSTED


def test_classify_ssl_error():
    err = Exception("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol")
    assert classify_error(err) == ErrorKind.SSL_ERROR


def test_classify_network_error():
    err = Exception("Connection refused")
    assert classify_error(err) == ErrorKind.NETWORK_ERROR


def test_classify_timeout():
    err = Exception("Request timeout after 30s")
    assert classify_error(err) == ErrorKind.NETWORK_ERROR


def test_classify_unknown():
    err = Exception("Something unexpected happened")
    assert classify_error(err) == ErrorKind.AGENT_ERROR


def test_ssl_is_retryable():
    assert ErrorKind.SSL_ERROR.is_retryable is True


def test_network_is_retryable():
    assert ErrorKind.NETWORK_ERROR.is_retryable is True


def test_quota_not_retryable():
    assert ErrorKind.QUOTA_EXHAUSTED.is_retryable is False


def test_quota_should_stop():
    assert ErrorKind.QUOTA_EXHAUSTED.should_stop_run is True


def test_ssl_should_not_stop():
    assert ErrorKind.SSL_ERROR.should_stop_run is False


def test_extract_retry_delay_from_message():
    err = Exception("retryDelay': '40s' something")
    assert _extract_retry_delay(err) == 40.0


def test_extract_retry_delay_natural():
    err = Exception("Please retry in 16.975412065s.")
    assert abs(_extract_retry_delay(err) - 16.975412065) < 0.01


def test_extract_retry_delay_default():
    err = Exception("some random error")
    assert _extract_retry_delay(err) == 60.0


def test_quota_exhausted_exception():
    e = QuotaExhaustedError(retry_after=40.0)
    assert e.retry_after == 40.0
    assert "40" in str(e)
