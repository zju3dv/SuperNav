from types import SimpleNamespace

import pytest

from supernav.runtime import image_prune_proxy


@pytest.mark.parametrize("upstream", [None, " \t "])
def test_proxy_requires_explicit_upstream(monkeypatch, capsys, upstream):
    if upstream is None:
        monkeypatch.delenv("HAB_BENCH_RELAY_UPSTREAM", raising=False)
    else:
        monkeypatch.setenv("HAB_BENCH_RELAY_UPSTREAM", upstream)

    def unexpected_config(_args):
        pytest.fail("Missing upstream must fail before creating an HTTP client")

    monkeypatch.setattr(image_prune_proxy, "_ProxyConfig", unexpected_config)
    with pytest.raises(SystemExit) as exc:
        image_prune_proxy.main([])
    assert exc.value.code == 2
    assert "--upstream or HAB_BENCH_RELAY_UPSTREAM" in capsys.readouterr().err


@pytest.mark.parametrize("use_cli", [False, True])
def test_proxy_uses_explicit_upstream_without_network(monkeypatch, use_cli):
    monkeypatch.setenv("HAB_BENCH_RELAY_UPSTREAM", "https://env.example.invalid/v1")
    observed = []

    def make_config(args):
        observed.append(args.upstream)
        return SimpleNamespace(
            upstream=args.upstream,
            keep_latest_items=args.keep_latest_items,
            client=SimpleNamespace(close=lambda: None),
        )

    server = SimpleNamespace(
        server_address=("127.0.0.1", 8993),
        serve_forever=lambda: None,
        server_close=lambda: None,
    )
    monkeypatch.setattr(image_prune_proxy, "_ProxyConfig", make_config)
    monkeypatch.setattr(image_prune_proxy, "ThreadingHTTPServer", lambda *_args: server)
    argv = ["--upstream", "https://cli.example.invalid/v1"] if use_cli else []
    assert image_prune_proxy.main(argv) == 0
    expected = "https://cli.example.invalid/v1" if use_cli else "https://env.example.invalid/v1"
    assert observed == [expected]
