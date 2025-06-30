import os
from github import Github
import requests
from agno.tools import tool
from agno.agent import Agent


@tool
def fetch_ci_failures(repo_name: str, pr_number: int) -> list:
    """
    Fetch CI failure data from GitHub API.

    Args:
        repo_name (str): Repository in owner/repo format (e.g., 'owner/repo')
        pr_number (int): Pull request number

    Returns:
        list: List of failure data containing job name, id and log information
    """
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

                    failure_data = {
                        "name": job.name,
                        "id": job_id,
                        "log": log_content,
                    }
                    failures.append(failure_data)

        return failures

        """
        Retrieve CI failure logs from agent's session state.
        
        Args:
            agent (Agent): The agent instance to access session state
            job_name (str, optional): Filter logs by job name
            
        Returns:
            list: List of failure logs matching the criteria
        """
        # Handle None session_state
        if agent.session_state is None or "ci_logs" not in agent.session_state:
            return []

        logs = agent.session_state["ci_logs"]
        if job_name:
            logs = [log for log in logs if log["name"] == job_name]

        return logs
