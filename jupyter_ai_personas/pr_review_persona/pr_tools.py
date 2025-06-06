import os
from typing import List
from agno.tools import Toolkit
from agno.utils.log import logger
import requests
import re
from unidiff import PatchSet
from io import StringIO


class PRTools(Toolkit):
    def __init__(self, **kwargs):
        self.github_token = os.getenv("GITHUB_TOKEN")
        if not self.github_token:
            logger.warning("GITHUB_TOKEN environment variable is not set. GitHub operations will be limited.")
        super().__init__(name="pr_tools", tools=[
            self.fetch_pr_data,
            self.extract_code_changes
        ], **kwargs)

    def fetch_pr_data(self, repo_url: str, pr_number: int) -> dict:
        """
        Fetch pull request data from GitHub API.
        
        Args:
            repo_url: GitHub repository URL
            pr_number: Pull request number
            
        Returns:
            dict: Processed pull request data containing files and changes
            
        Raises:
            ValueError: If GitHub URL is invalid, API response is unexpected, or GITHUB_TOKEN is not set
            requests.RequestException: If API request fails
        """

        if not self.github_token:
            raise ValueError("GITHUB_TOKEN environment variable is not set")
            
        if "${{" in str(self.github_token):
            logger.error("Invalid token format: Contains GitHub Actions syntax. Please use a plain token string.")
            raise ValueError("GitHub token contains GitHub Actions syntax (${{ ... }}). Please use a plain token string.")

        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

        match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
        if not match:
            raise ValueError("Invalid GitHub URL format")

        owner, repo = match.groups()
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
        
        try:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            files = resp.json()
            if not isinstance(files, list):
                raise ValueError("Unexpected API response format")
            
            processed_files = []
            for file in files:
                if isinstance(file, dict):
                    filename = file.get('filename', '')
                    patch = file.get('patch', '')
                    if patch:
                        patch = f"--- a/{filename}\n+++ b/{filename}\n{patch}"
                    
                    processed_files.append({
                        'filename': filename,
                        'status': file.get('status', ''),
                        'additions': file.get('additions', 0),
                        'deletions': file.get('deletions', 0),
                        'changes': file.get('changes', 0),
                        'patch': patch
                    })
            
            return {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "files": processed_files
            }
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch PR data: {str(e)}")
            raise


    @staticmethod
    def extract_code_changes(pr_data: dict) -> List[dict]:
        """
        Extract and process code changes from pull request data.
        
        Args:
            pr_data: Pull request data from fetch_pr_data
            
        Returns:
            list: List of file changes with detailed line information
        """
        file_diffs = []
        
        for file in pr_data["files"]:
            filename = file["filename"]
            patch = file.get("patch")
            
            if not patch:
                logger.info(f"No patch for {filename}, skipping.")
                continue
            
            file_change = {
                "file": filename,
                "changes": []
            }
            
            try:
                patchset = PatchSet(StringIO(patch))
                
                for patched_file in patchset:
                    for hunk in patched_file:
                        for line in hunk:
                            if line.is_added or line.is_removed or line.is_context:
                                file_change["changes"].append({
                                    "line_type": (
                                        "added" if line.is_added
                                        else "removed" if line.is_removed
                                        else "context"
                                    ),
                                    "line_number": (
                                        line.target_line_no if line.is_added or line.is_context 
                                        else line.source_line_no
                                    ),
                                    "content": line.value.strip()
                                })
                
                file_diffs.append(file_change)
                
            except Exception as e:
                logger.error(f"Error parsing patch for {filename}: {e}")
                continue
        
        return file_diffs
