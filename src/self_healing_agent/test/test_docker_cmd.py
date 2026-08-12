import docker
from unittest.mock import MagicMock, patch

from self_healing_agent.src.functions.docker_cmd import *

@patch("self_healing_agent.src.functions.docker_cmd.client")
def test_run_container_success(mock_client):
    mock_container = MagicMock()
    mock_container.logs.return_value = b"hello world"

    mock_client.containers.run.return_value = mock_container

    result = start_container("ubuntu", "echo hello")

    assert result == "hello world"
    mock_client.containers.run.assert_called_once_with(
        'ubuntu', name='echo hello', volumes=None, command=None, remove=None, detach=True
    )


@patch("self_healing_agent.src.functions.docker_cmd.client")
def test_run_container_image_not_found(mock_client, capsys):
    mock_client.containers.run.side_effect = docker.errors.ImageNotFound(
        "image not found"
    )

    result = start_container("bad-image", "echo hello")

    assert result is None

    captured = capsys.readouterr()
    assert "Image not found" in captured.out


@patch("self_healing_agent.src.functions.docker_cmd.client")
def test_delete_container_success(mock_client, capsys):
    mock_container = MagicMock()
    mock_client.containers.get.return_value = mock_container

    delete_container("container123")

    mock_container.remove.assert_called_once_with(force=True)

    captured = capsys.readouterr()
    assert "removed successfully" in captured.out


@patch("self_healing_agent.src.functions.docker_cmd.client")
def test_delete_container_not_found(mock_client, capsys):
    mock_client.containers.get.side_effect = docker.errors.NotFound(
        "container not found"
    )

    delete_container("container123")

    captured = capsys.readouterr()
    assert "not found" in captured.out


@patch("self_healing_agent.src.functions.docker_cmd.client")
def test_restart_container_success(mock_client):
    mock_container = MagicMock()
    mock_client.containers.get.return_value = mock_container

    restart_container("container123")

    mock_container.restart.assert_called_once()


@patch("self_healing_agent.src.functions.docker_cmd.client")
def test_scale_memory_success(mock_client):
    mock_container = MagicMock()
    mock_client.containers.get.return_value = mock_container

    update_memory("container123", "512m")

    mock_container.update.assert_called_once_with(mem_limit="512m", memswap_limit='512m')


@patch("self_healing_agent.src.functions.docker_cmd.client")
def test_rollback_container_success(mock_client):
    old_container = MagicMock()
    new_container = MagicMock()

    new_container.id = "new_container_id"

    mock_client.containers.get.return_value = old_container
    mock_client.containers.run.return_value = new_container

    result = rollback_container(
        "container123",
        "previous-image"
    )

    old_container.stop.assert_called_once()
    old_container.remove.assert_called_once()

    mock_client.containers.run.assert_called_once_with(
        "previous-image",
        detach=True,
    )

    assert result == "new_container_id"


@patch("self_healing_agent.src.functions.docker_cmd.client")
def test_rollback_container_not_found(mock_client, capsys):
    mock_client.containers.get.side_effect = docker.errors.NotFound(
        "container not found"
    )

    result = rollback_container(
        "container123",
        "previous-image"
    )

    assert result is None

    captured = capsys.readouterr()
    assert "not found" in captured.out