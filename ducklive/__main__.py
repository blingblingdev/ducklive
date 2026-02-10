"""DuckLive CLI entry point."""

import click


@click.group()
@click.version_option()
def cli():
    """DuckLive - Real-time face swap + voice change network webcam."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--port", default=8080, help="Dashboard port")
@click.option("--test", is_flag=True, help="Test mode (synthetic video/audio, no feed client needed)")
@click.option("--dev", is_flag=True, help="Development mode (verbose logging)")
def server(host: str, port: int, test: bool, dev: bool):
    """Start DuckLive server (runs on GPU machine, receives video from clients)."""
    from ducklive.model_check import check_models

    check_models(strict=False)

    from ducklive.server.app import start_server

    start_server(
        host=host,
        port=port,
        test_mode=test,
        dev=dev,
    )


@cli.command()
@click.option("--server-url", default=None, help="Server URL (auto-discovers if not set)")
@click.option("--no-camera", is_flag=True, help="Skip virtual camera output")
@click.option("--no-audio", is_flag=True, help="Skip virtual audio output")
@click.option("--webui-port", default=8081, help="Client Web UI port")
@click.option("--dev", is_flag=True, help="Development mode")
def client(server_url: str | None, no_camera: bool, no_audio: bool, webui_port: int, dev: bool):
    """Start DuckLive client (runs on Mac, receives stream)."""
    from ducklive.client.app import start_client

    start_client(
        server_url=server_url,
        enable_camera=not no_camera,
        enable_audio=not no_audio,
        webui_port=webui_port,
        dev=dev,
    )


@cli.command("check-models")
def check_models_cmd():
    """Check that all required models are present."""
    from ducklive.model_check import check_models

    if check_models(strict=True):
        from rich.console import Console
        Console().print("[green bold]All models found. Ready to go![/]")


if __name__ == "__main__":
    cli()
