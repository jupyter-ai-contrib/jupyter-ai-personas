import pytest
from unittest.mock import Mock, patch
import os
from github import Github
import requests


def fetch_ci_failures_logic(repo_name: str, pr_number: int) -> list:
    github_token = os.getenv("GITHUB_ACCESS_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_ACCESS_TOKEN environment variable is not set")

    g = Github(github_token)
    repo = g.get_repo(repo_name)
    pr_data = repo.get_pull(pr_number)
    runs = repo.get_workflow_runs(branch=pr_data.head.ref)
    failures = []

    for run in runs:
        if run.head_sha == pr_data.head.sha:
            jobs = run.jobs()

            for job in jobs:
                if job.conclusion == "failure":
                    job_id = job.raw_data["id"]

                    headers = {
                        "Accept": "application/vnd.github+json",
                        "Authorization": f"Bearer {github_token}",
                        "X-GitHub-Api-Version": "2022-11-28",
                    }
                    log_url = f"https://api.github.com/repos/{repo_name}/actions/jobs/{job_id}/logs"
                    log_response = requests.get(log_url, headers=headers)

                    if log_response.status_code != 200:
                        raise Exception(
                            f"Failed to fetch logs: {log_response.status_code} {log_response.text}"
                        )

                    log_content = log_response.text

                    failure_data = {"name": job.name, "id": job_id, "log": log_content}
                    failures.append(failure_data)

        return failures


def test_fetch_without_token():
    with patch("os.getenv", return_value=None):
        with pytest.raises(
            ValueError, match="GITHUB_ACCESS_TOKEN environment variable is not set"
        ):
            fetch_ci_failures_logic("owner/repo", 123)


@patch("github.Requester.Requester.requestJsonAndCheck")
def test_fetch_ci_failures_no_failures(mock_request):
    mock_request.side_effect = [
        ({}, {"id": 123}),
        ({}, {"head": {"ref": "main", "sha": "test_sha"}}),
        (
            {},
            {
                "workflow_runs": [
                    {
                        "id": 456,
                        "head_sha": "test_sha",
                        "url": "https://api.github.com/repos/owner/repo/actions/runs/456",
                        "jobs_url": "https://api.github.com/repos/owner/repo/actions/runs/456/jobs",
                    }
                ]
            },
        ),
        ({}, {"jobs": []}),
    ]

    with patch("os.getenv", return_value="dummy_token"):
        failures = fetch_ci_failures_logic("owner/repo", 123)
        assert len(failures) == 0


@patch("github.Requester.Requester.requestJsonAndCheck")
def test_fetch_ci_failures_with_failures(mock_request):
    mock_request.side_effect = [
        ({}, {"id": 123}),
        ({}, {"head": {"ref": "main", "sha": "test_sha"}}),
        (
            {},
            {
                "workflow_runs": [
                    {
                        "id": 456,
                        "head_sha": "test_sha",
                        "url": "https://api.github.com/repos/owner/repo/actions/runs/456",
                        "jobs_url": "https://api.github.com/repos/owner/repo/actions/runs/456/jobs",
                    }
                ]
            },
        ),
        (
            {},
            {
                "jobs": [
                    {
                        "conclusion": "failure",
                        "name": "test_job",
                        "id": "123",
                        "url": "https://api.github.com/repos/owner/repo/actions/jobs/123",
                    }
                ]
            },
        ),
        (
            {},
            {
                "conclusion": "failure",
                "name": "test_job",
                "id": "123",
                "url": "https://api.github.com/repos/owner/repo/actions/jobs/123",
            },
        ),
    ]

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "line1\nline2\nerror: test error\nline4\nline5"

    with (
        patch("os.getenv", return_value="dummy_token"),
        patch("requests.get", return_value=mock_response),
    ):
        failures = fetch_ci_failures_logic("owner/repo", 123)

        assert len(failures) == 1
        assert failures[0]["name"] == "test_job"
        assert failures[0]["id"] == "123"
        assert "log" in failures[0]
        assert "error: test error" in failures[0]["log"]
