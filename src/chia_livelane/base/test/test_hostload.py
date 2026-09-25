"""The load guard must warn, never block, and never lie about what it covers."""
from __future__ import annotations

from chia_livelane.base import hostload
from chia_livelane.base.hostload import HostLoad, host_load, load_warning


class TestHostLoad:
    def test_reports_cpus_and_load(self) -> None:
        hl = host_load()
        assert hl.cpus is None or hl.cpus >= 1
        if hl.load1 is not None:
            assert hl.load1 >= 0.0

    def test_per_cpu_is_none_without_data(self) -> None:
        assert HostLoad().per_cpu is None
        assert HostLoad(load1=4.0, cpus=None).per_cpu is None

    def test_per_cpu_divides_by_cores(self) -> None:
        assert HostLoad(load1=12.0, cpus=24).per_cpu == 0.5

    def test_as_dict_omits_missing_fields(self) -> None:
        # A JSON record must not carry `"load1": null`, a reader that treats
        # a null as zero would conclude the box was idle.
        assert "load1" not in HostLoad(cpus=8).as_dict()
        assert HostLoad(cpus=8).as_dict() == {"cpus": 8}


class TestLoadWarning:
    def test_quiet_machine_gets_no_warning(self, monkeypatch) -> None:
        monkeypatch.setattr(hostload, "host_load",
                            lambda: HostLoad(0.4, 0.4, 0.4, 24))
        assert load_warning() is None

    def test_busy_machine_is_warned_about(self, monkeypatch) -> None:
        monkeypatch.setattr(hostload, "host_load",
                            lambda: HostLoad(48.0, 44.0, 33.0, 24))
        w = load_warning()
        assert w and "24 cores" in w

    def test_warning_says_what_is_NOT_affected(self, monkeypatch) -> None:
        # The whole point is that a soundness run on a loaded box is still
        # valid. A warning that implied otherwise would cause good results to
        # be thrown away and re-run for nothing.
        monkeypatch.setattr(hostload, "host_load",
                            lambda: HostLoad(48.0, 44.0, 33.0, 24))
        w = load_warning() or ""
        assert "reuse" in w and "unaffected" in w

    def test_threshold_is_per_core_not_absolute(self, monkeypatch) -> None:
        # Load 6 is quiet on 24 cores and saturated on 8. An absolute
        # threshold would be wrong on one of the two.
        monkeypatch.setattr(hostload, "host_load",
                            lambda: HostLoad(6.0, 6.0, 6.0, 24))
        assert load_warning() is None, "0.25 per core is a quiet machine"
        monkeypatch.setattr(hostload, "host_load",
                            lambda: HostLoad(6.0, 6.0, 6.0, 8))
        assert load_warning() is not None, "0.75 per core is not"

    def test_threshold_boundary_warns(self, monkeypatch) -> None:
        # Exactly at the threshold warns. The tie is broken toward warning
        # because a spurious warning costs one line and a missed one costs a
        # timing nobody knows to distrust.
        monkeypatch.setattr(hostload, "host_load",
                            lambda: HostLoad(12.0, 12.0, 12.0, 24))
        assert load_warning(threshold=0.5) is not None
        assert load_warning(threshold=0.51) is None

    def test_never_raises_when_load_is_unavailable(self, monkeypatch) -> None:
        monkeypatch.setattr(hostload, "host_load", lambda: HostLoad(cpus=None))
        assert load_warning() is None
