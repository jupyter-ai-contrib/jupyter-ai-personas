import os
import json
from agno.tools import Toolkit
from agno.utils.log import logger
import re
from github import Github
import requests
from agno.agent import Agent

class CITools(Toolkit):
    def __init__(self, **kwargs):
        self.github_token = os.getenv("GITHUB_ACCESS_TOKEN")

        if not self.github_token:
            logger.warning("GITHUB_ACCESS_TOKEN environment variable is not set. GitHub operations will be limited.")
        super().__init__(name="ci_tools", tools=[
            self.fetch_ci_failure_data,
            self.get_ci_logs
        ],  **kwargs)

    async def fetch_ci_failure_data(self, agent: Agent,  repo_url: str, pr_number: int) -> list:
        """
        Fetch CI Failure data from GitHub API and store it in the agent's session state.
        
        Args:
            agent (Agent): The agent instance to store logs in session state
            repo_url (str): URL of the GitHub repository
            pr_number (int): Pull request number
            
        Returns:
            list: List of failure data containing job name, id and log information
        """
        match = None
        if "github.com" in repo_url:
            match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
            
        if not match:
            raise ValueError("Invalid GitHub URL format. Expected either github.com/owner/repo or api.github.com/repos/owner/repo")

        owner, repo_name = match.groups()
        repo_name = f"{owner}/{repo_name}"

        g = Github(os.getenv("GITHUB_ACCESS_TOKEN"))
        repo = g.get_repo(repo_name)
        pr_data = repo.get_pull(pr_number)

        runs = repo.get_workflow_runs(branch=pr_data.head.ref)
        failures = []

        for run in runs:
            if run.head_sha == pr_data.head.sha:
                print("check1")
                print(repo_name)
                print(run.id)

                jobs = run.jobs()

                for job in jobs:
                    if job.conclusion == "failure":
                        print("check2")
                        print(f"Found failed job: {job.name}")

                        job_id = job.raw_data["id"]
                        

                        headers = {
                            "Accept": "application/vnd.github+json",
                            "Authorization": f"Bearer {self.github_token}",
                            "X-GitHub-Api-Version": "2022-11-28"
                        }
                        log_url = f"https://api.github.com/repos/{repo_name}/actions/jobs/{job_id}/logs"
                        log_response = requests.get(log_url, headers=headers)
                        
                        if log_response.status_code != 200:
                            raise Exception(f"Failed to fetch logs: {log_response.status_code} {log_response.text} from {log_url}")
                        log_content = log_response.text

                        ##If seeing ThrottlingException in test aws account uncomment these lines [81-87] and line  92 AND comment line 93

                        # # Extract key error lines from the log
                        # log_lines = log_content.splitlines()
                        # error_lines = []
                        # for line in log_lines[-20:]:  
                        #     if 'error:' in line.lower() or 'fail:' in line.lower():
                        #         error_lines.append(line)
                        #         if len(error_lines) >= 10: 
                        #             break
                        
                        failure_data = {
                            "name": job.name,
                            "id": job_id,
                            # "error_lines": error_lines if error_lines else [log_lines[-1]],  
                            "log": log_content
                        }
                        failures.append(failure_data)

                        if agent.session_state is None:
                            agent.session_state = {}
                        if "ci_logs" not in agent.session_state:
                            agent.session_state["ci_logs"] = []
                        
                        agent.session_state["ci_logs"].append(failure_data)

        print(f"Found {len(failures)} failed jobs")
        return failures

    async def get_ci_logs(self, agent: Agent, job_name: str = None) -> list:
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