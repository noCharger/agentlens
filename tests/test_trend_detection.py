"""Tests for trend detection and anomaly alerting."""

from agentlens.core.trend_detection import (
    analyze_metric_trend,
    detect_anomaly,
    detect_declining_trend,
    detect_rate_change,
)


class TestDecliningTrend:
    def test_no_trend_stable(self):
        values = [80.0, 80.0, 80.0, 80.0, 80.0]
        alert = detect_declining_trend(values)
        assert alert is None

    def test_declining_trend_detected(self):
        values = [90.0, 85.0, 78.0, 70.0, 60.0]
        alert = detect_declining_trend(values, decline_threshold=-2.0)
        assert alert is not None
        assert alert.alert_type == "declining_trend"

    def test_too_few_values(self):
        values = [80.0, 70.0]
        alert = detect_declining_trend(values, window_size=5)
        assert alert is None

    def test_rising_trend_no_alert(self):
        values = [60.0, 70.0, 75.0, 80.0, 85.0]
        alert = detect_declining_trend(values)
        assert alert is None


class TestAnomaly:
    def test_no_anomaly(self):
        values = [80.0, 81.0, 79.0, 80.0, 80.5]
        alert = detect_anomaly(values)
        assert alert is None

    def test_anomaly_detected_low(self):
        values = [80.0, 81.0, 79.0, 80.0, 40.0]
        alert = detect_anomaly(values, z_threshold=2.0)
        assert alert is not None
        assert alert.alert_type == "anomaly"
        assert "below" in alert.message

    def test_anomaly_detected_high(self):
        values = [80.0, 81.0, 79.0, 80.0, 120.0]
        alert = detect_anomaly(values, z_threshold=2.0)
        assert alert is not None
        assert "above" in alert.message

    def test_too_few_values(self):
        values = [80.0, 40.0]
        alert = detect_anomaly(values)
        assert alert is None


class TestRateChange:
    def test_no_drop(self):
        values = [80.0, 80.0]
        alert = detect_rate_change(values)
        assert alert is None

    def test_drop_detected(self):
        values = [85.0, 70.0]
        alert = detect_rate_change(values, drop_threshold=10.0)
        assert alert is not None
        assert alert.alert_type == "rate_change"

    def test_small_drop_below_threshold(self):
        values = [85.0, 80.0]
        alert = detect_rate_change(values, drop_threshold=10.0)
        assert alert is None

    def test_critical_severity_on_large_drop(self):
        values = [90.0, 50.0]
        alert = detect_rate_change(values, drop_threshold=10.0)
        assert alert is not None
        assert alert.severity.value == "critical"


class TestFullAnalysis:
    def test_stable_series(self):
        values = [80.0, 81.0, 79.0, 80.0, 80.5]
        analysis = analyze_metric_trend(values)
        assert analysis.trend_direction == "stable"
        assert len(analysis.alerts) == 0

    def test_falling_series_with_alerts(self):
        values = [90.0, 85.0, 78.0, 70.0, 60.0]
        analysis = analyze_metric_trend(values)
        assert analysis.trend_direction == "falling"
        assert len(analysis.alerts) > 0
