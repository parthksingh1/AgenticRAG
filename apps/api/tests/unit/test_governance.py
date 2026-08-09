"""Governance: SSRF defence, webhook signing, drift, audit redaction, retention.

The SSRF and signature tests are the security-relevant ones here and are written
as attacks rather than as happy paths: each case is a way someone would actually
try to get an outbound request pointed somewhere it should not go, or to forge a
delivery.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st
from src.core import net
from src.core.errors import ValidationFailedError
from src.governance import audit, drift, retention
from src.services import webhooks


class TestSsrfGuard:
    """Outbound-request validation."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/admin",
            "http://localhost/",
            "https://10.0.0.5/internal",
            "http://192.168.1.1/",
            "http://172.16.0.1/",
            # The cloud metadata endpoint: the highest-value SSRF target there
            # is, because it hands out the instance's credentials.
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            "http://[::1]/",
            # Loopback written so no string denylist would catch it.
            "http://0x7f.1/",
            "http://2130706433/",
            # IPv4-mapped IPv6, which a hand-rolled range check misses.
            "http://[::ffff:127.0.0.1]/",
        ],
    )
    def test_private_and_loopback_targets_are_refused(self, url: str) -> None:
        """Every form of "inside the perimeter" is rejected, however written."""
        with pytest.raises(ValidationFailedError):
            net.validate_public_url(url)

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x/", "ftp://example.com/"])
    def test_non_http_schemes_are_refused(self, url: str) -> None:
        """file:// reads the container's disk; gopher:// smuggles protocols."""
        with pytest.raises(ValidationFailedError, match="scheme"):
            net.validate_public_url(url)

    def test_internal_service_ports_are_refused(self) -> None:
        """Not sufficient on its own, but it removes the obvious cases."""
        with pytest.raises(ValidationFailedError, match="port"):
            net.validate_public_url("http://example.com:6379/", resolve=False)

    def test_an_absurdly_long_url_is_refused(self) -> None:
        """A URL past the limit is a payload, not an endpoint."""
        with pytest.raises(ValidationFailedError, match="longer than"):
            net.validate_public_url("https://example.com/" + "a" * 3000, resolve=False)

    def test_a_url_with_no_host_is_refused(self) -> None:
        """There is nothing to validate."""
        with pytest.raises(ValidationFailedError, match="no host"):
            net.validate_public_url("http:///path", resolve=False)

    def test_a_name_that_does_not_resolve_is_refused(self) -> None:
        """Better to reject at registration than fail every delivery later."""
        with pytest.raises(ValidationFailedError, match="does not resolve"):
            net.validate_public_url("https://this-name-does-not-exist.invalid/hook")

    def test_a_public_address_is_accepted(self) -> None:
        """The guard must not reject the legitimate case."""
        target = net.validate_public_url("https://93.184.216.34/hook")
        assert target.host == "93.184.216.34"
        assert target.port == 443
        assert target.addresses == ("93.184.216.34",)

    def test_the_default_port_follows_the_scheme(self) -> None:
        """A bare https URL is port 443, not 80."""
        assert net.validate_public_url("http://93.184.216.34/", resolve=False).port == 80
        assert net.validate_public_url("https://93.184.216.34/", resolve=False).port == 443


class TestWebhookSignatures:
    """HMAC signing and verification."""

    def test_a_signature_verifies_against_its_own_body(self) -> None:
        """The happy path."""
        body = b'{"event":"answer.completed"}'
        header = webhooks.sign(body, secret="k")
        assert webhooks.verify(body, header=header, secret="k")

    def test_a_modified_body_fails(self) -> None:
        """Tampering with the payload invalidates the signature."""
        header = webhooks.sign(b'{"amount":1}', secret="k")
        assert not webhooks.verify(b'{"amount":1000}', header=header, secret="k")

    def test_the_wrong_secret_fails(self) -> None:
        """A signature is only as good as the key it was made with."""
        header = webhooks.sign(b"{}", secret="right")
        assert not webhooks.verify(b"{}", header=header, secret="wrong")

    def test_a_stale_signature_is_refused(self) -> None:
        """This is why the timestamp is inside the signed input.

        Without it a captured delivery stays valid forever and can be replayed
        indefinitely.
        """
        old = webhooks.sign(b"{}", secret="k", timestamp=int(time.time()) - 4000)
        assert not webhooks.verify(b"{}", header=old, secret="k")
        assert webhooks.verify(b"{}", header=old, secret="k", tolerance_seconds=100_000)

    def test_a_replayed_timestamp_with_a_fresh_body_fails(self) -> None:
        """Reusing a valid timestamp does not let a different body through."""
        header = webhooks.sign(b'{"a":1}', secret="k")
        assert not webhooks.verify(b'{"a":2}', header=header, secret="k")

    @pytest.mark.parametrize("header", ["", "garbage", "t=abc,v1=xx", "v1=onlysig", "t=,v1="])
    def test_a_malformed_header_is_refused_rather_than_raising(self, header: str) -> None:
        """A receiver must not be crashable by a malformed signature header."""
        assert not webhooks.verify(b"{}", header=header, secret="k")

    def test_secrets_are_high_entropy_and_distinct(self) -> None:
        """Two mints must never collide."""
        assert webhooks.generate_secret() != webhooks.generate_secret()
        assert len(webhooks.generate_secret()) > 40

    def test_secret_hashing_is_deterministic_and_sensitive(self) -> None:
        """Same input, same hash; one character different, different hash."""
        assert webhooks.hash_secret("s") == webhooks.hash_secret("s")
        assert webhooks.hash_secret("s") != webhooks.hash_secret("t")

    def test_an_oversized_payload_is_replaced_with_a_pointer(self) -> None:
        """A webhook is a notification, not a transport for a conversation."""
        payload = {"id": "msg_1", "type": "answer", "content": "x" * 200_000}
        truncated = webhooks._truncate(payload)
        assert truncated["truncated"] is True
        assert truncated["id"] == "msg_1"
        assert "content" not in truncated

    def test_a_normal_payload_passes_through_unchanged(self) -> None:
        """Truncation must not touch the ordinary case."""
        payload = {"id": "msg_1", "content": "short"}
        assert webhooks._truncate(payload) == payload

    def test_the_backoff_schedule_is_monotonic(self) -> None:
        """Retries must back off, not oscillate."""
        delays = webhooks.RETRY_DELAYS_SECONDS
        assert list(delays) == sorted(delays)
        assert delays[0] > 0


class TestDrift:
    """Distribution comparison."""

    def test_identical_distributions_have_no_drift(self) -> None:
        """A stable corpus reports stable."""
        bins = drift.histogram([0.1, 0.5, 0.9] * 10)
        assert drift.psi(bins, bins) == 0.0
        assert drift.severity(drift.psi(bins, bins)) == "stable"

    def test_a_reversed_distribution_is_material(self) -> None:
        """Scores moving from the top of the range to the bottom must alert."""
        high = drift.histogram([0.95] * 50)
        low = drift.histogram([0.05] * 50)
        assert drift.severity(drift.psi(high, low)) == "material"

    def test_an_empty_bin_does_not_produce_infinity(self) -> None:
        """Without the epsilon a single empty bin sends PSI to infinity.

        Every comparison would then alert, which is the same as no alerting.
        """
        value = drift.psi([1.0, 0.0], [0.0, 1.0])
        assert value == value  # not NaN
        assert value < float("inf")

    def test_mismatched_bin_counts_are_rejected(self) -> None:
        """Comparing them would truncate one side and report a made-up number."""
        with pytest.raises(ValueError, match="same bin count"):
            drift.psi([0.5, 0.5], [0.3, 0.3, 0.4])

    def test_a_histogram_sums_to_one(self) -> None:
        """It is a distribution, not a count."""
        bins = drift.histogram([0.1, 0.2, 0.9, 0.95])
        assert sum(bins) == pytest.approx(1.0)

    def test_values_above_the_range_land_in_the_last_bin(self) -> None:
        """A score that has moved out of range is the signal, not an error."""
        assert drift.histogram([5.0], bins=2, upper=1.0) == [0.0, 1.0]

    def test_an_empty_series_gives_empty_bins(self) -> None:
        """No samples means no distribution, not a crash."""
        assert drift.histogram([], bins=3) == [0.0, 0.0, 0.0]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, "stable"),
            (0.09, "stable"),
            (0.1, "moderate"),
            (0.24, "moderate"),
            (0.25, "material"),
        ],
    )
    def test_the_severity_thresholds_are_the_documented_ones(
        self, value: float, expected: str
    ) -> None:
        """The conventional PSI cutoffs, held to exactly at the boundary."""
        assert drift.severity(value) == expected

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=100))
    def test_psi_against_itself_is_always_zero(self, values: list[float]) -> None:
        """Whatever the distribution, it has not drifted from itself."""
        bins = drift.histogram(values)
        assert drift.psi(bins, bins) == 0.0

    def test_the_metric_series_bounds_differ_per_metric(self) -> None:
        """Binning a 400-character query against [0, 1] would report stability forever."""
        _, score_upper = drift._series([], "retrieval_top_score")
        _, length_upper = drift._series([], "query_length")
        assert score_upper == 1.0
        assert length_upper > 1.0


class TestAuditRedaction:
    """What never reaches the audit log."""

    @pytest.mark.parametrize(
        "key",
        ["secret", "api_key", "password", "clerk_secret_key", "key_hash", "REFRESH_TOKEN"],
    )
    def test_secret_shaped_keys_are_redacted(self, key: str) -> None:
        """The audit log is read by more people than the database is."""
        assert audit.redact_secrets({key: "sensitive"})[key] == audit.REDACTED

    def test_redaction_reaches_nested_objects(self) -> None:
        """Config blobs nest, and a secret one level down is still a secret."""
        result = audit.redact_secrets({"model_policy": {"openai_api_key": "sk-live"}})
        assert result["model_policy"]["openai_api_key"] == audit.REDACTED

    def test_ordinary_values_survive(self) -> None:
        """Redaction that eats the diff makes the log useless."""
        payload = {"name": "Acme", "scopes": ["read"], "retention_days": 30}
        assert audit.redact_secrets(payload) == payload

    def test_the_diff_shows_only_what_changed(self) -> None:
        """A whole-object dump makes the reader diff two blobs by eye."""
        result = audit.diff({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 99, "c": 3})
        assert result == {"b": {"from": 2, "to": 99}}

    def test_the_diff_reports_added_and_removed_keys(self) -> None:
        """An appearing or vanishing field is a change worth recording."""
        assert audit.diff({}, {"new": 1}) == {"new": {"from": None, "to": 1}}
        assert audit.diff({"gone": 1}, {}) == {"gone": {"from": 1, "to": None}}

    def test_identical_states_produce_no_diff(self) -> None:
        """A no-op update should not look like a change."""
        assert audit.diff({"a": 1}, {"a": 1}) == {}


class TestRetention:
    """Retention windows."""

    def test_the_cutoff_is_the_window_before_now(self) -> None:
        """Thirty days back from the last day of March is the first."""
        cutoff = retention.cutoff_for(30, now=datetime(2026, 3, 31, tzinfo=UTC))
        assert cutoff.date().isoformat() == "2026-03-01"

    def test_no_policy_means_no_cutoff(self) -> None:
        """Unlimited retention must not be read as "delete everything"."""
        assert retention.cutoff_for(None) is None

    @given(st.integers(min_value=1, max_value=3650))
    def test_a_longer_window_always_means_an_earlier_cutoff(self, days: int) -> None:
        """Keeping data longer can never expire more of it."""
        now = datetime(2026, 6, 1, tzinfo=UTC)
        assert retention.cutoff_for(days + 1, now=now) < retention.cutoff_for(days, now=now)
