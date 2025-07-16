import pytest
from unittest.mock import Mock, patch, AsyncMock
from jupyter_ai_personas.pr_review_persona.persona import PRReviewPersona
from jupyterlab_chat.models import Message
from agno.team.team import Team
import asyncio
from dataclasses import asdict


class AwaitableAsyncMock(AsyncMock):
    """AsyncMock that properly handles awaiting coroutines."""

    async def __call__(self, *args, **kwargs):
        args = [await arg if asyncio.iscoroutine(arg) else arg for arg in args]
        kwargs = {
            k: await v if asyncio.iscoroutine(v) else v for k, v in kwargs.items()
        }
        return await super().__call__(*args, **kwargs)


@pytest.fixture
async def pr_persona():
    # Mock initialization arguments
    mock_ychat = AsyncMock()
    mock_manager = AsyncMock()
    mock_manager.outdated_timeout = 30000

    with patch(
        "jupyter_ai.personas.persona_awareness.PersonaAwareness"
    ) as mock_awareness:
        awareness_instance = AsyncMock()
        mock_awareness.return_value = awareness_instance

        # to prevent task creation
        async def mock_heartbeat():
            return

        awareness_instance._start_heartbeat = mock_heartbeat
        awareness_instance.outdated_timeout = 30000

        mock_config_manager = Mock()
        mock_config_manager.lm_provider.name = "test_provider"
        mock_config_manager.lm_provider_params = {"model_id": "test_model"}
        mock_log = Mock()
        mock_ychat.set_user = AwaitableAsyncMock()
        awareness_instance.set_local_state = AwaitableAsyncMock()
        awareness_instance.set_local_state_field = AwaitableAsyncMock()

        persona = PRReviewPersona(
            ychat=mock_ychat,
            manager=mock_manager,
            config_manager=mock_config_manager,
            log=mock_log,
            message_interrupted=False,
        )

        async def dummy_heartbeat():
            return

        awareness_instance._heartbeat_task = asyncio.create_task(dummy_heartbeat())

        try:
            yield persona
        finally:
            if not awareness_instance._heartbeat_task.done():
                awareness_instance._heartbeat_task.cancel()
            await asyncio.sleep(0)


@pytest.fixture
def mock_message():
    message = Mock(spec=Message)
    message.body = "Please review PR #123 in repo owner/repo"
    return message


@patch("jupyter_ai_personas.pr_review_persona.persona.Team")
@patch("agno.tools.github.GithubTools.authenticate")
@patch("boto3.Session")
@pytest.mark.asyncio
async def test_initialize_team(
    mock_boto_session, mock_github_auth, mock_team_class, pr_persona
):
    async for persona in pr_persona:
        mock_github_auth.return_value = Mock()
        mock_boto_session.return_value = Mock()

        mock_team = Mock()
        mock_code_quality = Mock()
        mock_code_quality.name = "code_quality"
        mock_documentation_checker = Mock()
        mock_documentation_checker.name = "documentation_checker"
        mock_security_checker = Mock()
        mock_security_checker.name = "security_checker"
        mock_github = Mock()
        mock_github.name = "github"

        mock_team.members = [
            mock_code_quality,
            mock_documentation_checker,
            mock_security_checker,
            mock_github,
        ]
        mock_team_class.return_value = mock_team

        with patch("os.getenv", return_value="dummy_token"):
            team = persona.initialize_team("test prompt")

        assert team is mock_team
        assert len(team.members) == 4

        mock_team_class.assert_called_once()

        mock_github_auth.assert_called()


@pytest.mark.asyncio
async def test_process_message_success(pr_persona, mock_message):
    async for persona in pr_persona:
        persona.send_message = AsyncMock()

        mock_history = AsyncMock()
        mock_history.aget_messages.return_value = []
        mock_response = Mock()
        mock_response.content = "PR review completed successfully"

        mock_team = Mock()
        mock_team.run.return_value = mock_response

        with (
            patch(
                "jupyter_ai_personas.pr_review_persona.persona.YChatHistory",
                return_value=mock_history,
            ),
            patch.object(persona, "initialize_team", return_value=mock_team),
        ):
            await persona.process_message(mock_message)

            assert persona.initialize_team.called
            assert mock_team.run.called
            assert persona.send_message.called


@pytest.mark.asyncio
async def test_process_message_value_error(pr_persona, mock_message):
    async for persona in pr_persona:
        persona.send_message = AsyncMock()

        mock_history = AsyncMock()
        mock_history.aget_messages.return_value = []

        mock_team = Mock()
        mock_team.run.side_effect = ValueError("Test error")

        with (
            patch(
                "jupyter_ai_personas.pr_review_persona.persona.YChatHistory",
                return_value=mock_history,
            ),
            patch.object(persona, "initialize_team", return_value=mock_team),
        ):
            await persona.process_message(mock_message)

            call_args = persona.send_message.call_args[0][0]
            assert "Configuration Error" in call_args
            assert "Test error" in call_args


@pytest.mark.asyncio
async def test_process_message_boto_error(pr_persona, mock_message):
    async for persona in pr_persona:
        persona.send_message = AsyncMock()

        mock_history = AsyncMock()
        mock_history.aget_messages.return_value = []
        from boto3.exceptions import Boto3Error

        mock_team = Mock()
        mock_team.run.side_effect = Boto3Error("AWS error")

        with (
            patch(
                "jupyter_ai_personas.pr_review_persona.persona.YChatHistory",
                return_value=mock_history,
            ),
            patch.object(persona, "initialize_team", return_value=mock_team),
        ):
            await persona.process_message(mock_message)

            call_args = persona.send_message.call_args[0][0]
            assert "PR Review Error" in call_args
            assert "AWS error" in call_args


@pytest.mark.asyncio
async def test_process_message_general_exception(pr_persona, mock_message):
    async for persona in pr_persona:
        persona.send_message = AsyncMock()

        mock_history = AsyncMock()
        mock_history.aget_messages.return_value = []

        mock_team = Mock()
        mock_team.run.side_effect = Exception("General error")

        with (
            patch(
                "jupyter_ai_personas.pr_review_persona.persona.YChatHistory",
                return_value=mock_history,
            ),
            patch.object(persona, "initialize_team", return_value=mock_team),
        ):
            await persona.process_message(mock_message)

            call_args = persona.send_message.call_args[0][0]
            assert "PR Review Error" in call_args
            assert "General error" in call_args
