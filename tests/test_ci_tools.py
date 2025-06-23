import pytest
from unittest.mock import Mock, patch, AsyncMock
from jupyter_ai_personas.pr_review_persona.ci_tools import CITools
from agno.agent import Agent
from github import Auth

@pytest.fixture
def ci_tools():
    tools = CITools()
    tools.github_token = 'dummy_token'
    return tools

@pytest.fixture
def mock_agent():
    agent = AsyncMock(spec=Agent)
    agent.session_state = {}
    return agent

@pytest.mark.asyncio
async def test_init_without_token():
    with patch('os.getenv', return_value=None):
        tools = CITools()
        assert tools.github_token is None

@pytest.mark.asyncio
async def test_init_with_token():
    with patch('os.getenv', return_value='test_token'):
        tools = CITools()
        assert tools.github_token == 'test_token'

@pytest.mark.asyncio
async def test_fetch_ci_failure_data_invalid_url(ci_tools, mock_agent):
    with pytest.raises(ValueError) as exc_info:
        ci_tools.fetch_ci_failure_data(mock_agent, "invalid_url", 123)
    assert "Invalid GitHub URL format" in str(exc_info.value)

@patch('github.Requester.Requester.requestJsonAndCheck')
@pytest.mark.asyncio
async def test_fetch_ci_failure_data_no_failures(mock_request, ci_tools, mock_agent):
   
    mock_request.side_effect = [
        ({}, {"id": 123}),  
        ({}, {"head": {"ref": "main", "sha": "test_sha"}}), 
        ({}, {"workflow_runs": [{"id": 456, "head_sha": "test_sha", "url": "https://api.github.com/repos/owner/repo/actions/runs/456", "jobs_url": "https://api.github.com/repos/owner/repo/actions/runs/456/jobs"}]}),  
        ({}, {"jobs": []}) 
    ]
    
    # Set GitHub token and execute test
    ci_tools.github_token = 'dummy_token'
    failures = ci_tools.fetch_ci_failure_data(mock_agent, "github.com/owner/repo", 123)
    assert len(failures) == 0
    assert "ci_logs" not in mock_agent.session_state

@patch('github.Requester.Requester.requestJsonAndCheck')
@pytest.mark.asyncio
async def test_fetch_ci_failure_data_with_failures(mock_request, ci_tools, mock_agent):
    # Mock GitHub API responses
    mock_request.side_effect = [
        ({}, {"id": 123}),  
        ({}, {"head": {"ref": "main", "sha": "test_sha"}}),  
        ({}, {"workflow_runs": [{"id": 456, "head_sha": "test_sha", "url": "https://api.github.com/repos/owner/repo/actions/runs/456", "jobs_url": "https://api.github.com/repos/owner/repo/actions/runs/456/jobs"}]}),  
        ({}, {"jobs": [{"conclusion": "failure", "name": "test_job", "id": "123", "url": "https://api.github.com/repos/owner/repo/actions/jobs/123"}]}), 
        ({}, {"conclusion": "failure", "name": "test_job", "id": "123", "url": "https://api.github.com/repos/owner/repo/actions/jobs/123"})  
    ]
    
    # Mock log response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "line1\nline2\nerror: test error\nline4\nline5"

    ci_tools.github_token = 'dummy_token'
    with patch('requests.get', return_value=mock_response):
        failures = ci_tools.fetch_ci_failure_data(mock_agent, "github.com/owner/repo", 123)
        
        assert len(failures) == 1
        assert failures[0]["name"] == "test_job"
        assert failures[0]["id"] == "123"
        assert "log" in failures[0]
        assert "error: test error" in failures[0]["log"]

@pytest.mark.asyncio
async def test_get_ci_logs_no_state(ci_tools, mock_agent):
    mock_agent.session_state = None
    logs = ci_tools.get_ci_logs(mock_agent)
    assert logs == []

@pytest.mark.asyncio
async def test_get_ci_logs_with_filter(ci_tools, mock_agent):
    mock_agent.session_state = {
        "ci_logs": [
            {"name": "job1", "log": "error1"},
            {"name": "job2", "log": "error2"}
        ]
    }
    logs = ci_tools.get_ci_logs(mock_agent, "job1")
    assert len(logs) == 1
    assert logs[0]["name"] == "job1"

@pytest.mark.asyncio
async def test_get_ci_logs_without_filter(ci_tools, mock_agent):
    mock_agent.session_state = {
        "ci_logs": [
            {"name": "job1", "log": "error1"},
            {"name": "job2", "log": "error2"}
        ]
    }
    logs = ci_tools.get_ci_logs(mock_agent)
    assert len(logs) == 2
