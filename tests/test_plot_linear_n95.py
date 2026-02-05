from __future__ import annotations

import plot_linear_n95


def test_plot_linearity_n95_creates_output(tmp_path, monkeypatch) -> None:
    def fake_n95(*_args, **_kwargs):
        return [0, 1], [2.0, 3.0]

    monkeypatch.setattr(plot_linear_n95, "load_linearity_n95", fake_n95)
    monkeypatch.chdir(tmp_path)

    plot_linear_n95.plot_linearity_n95()
    assert (tmp_path / "plots" / "linearity_n95_combined.pdf").exists()
