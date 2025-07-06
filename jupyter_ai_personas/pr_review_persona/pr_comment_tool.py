from typing import Any, Dict, List
from github import Github, GithubException
from os import getenv
from agno.tools import tool

@tool
def create_inline_pr_comments(repo_name: str, pr_number: int, comments: List[Dict[str, Any]]) -> str:
    """Create multiple inline comments on a pull request.
    
    Args:
        repo_name (str): The full name of the repository (e.g., 'owner/repo').
        pr_number (int): The number of the pull request.
        comments (List[Dict]): List of comment objects with the following structure:
            [
                {
                    "path": "path/to/file.py",  # Relative file path
                    "position": 10,            # Line number in the file
                    "body": "Comment text"      # The comment text
                },
                ...
            ]
    
    Returns:
        str: Success message with URLs of created comments or error.
    """
    print(f"[DEBUG] create_inline_pr_comments called with repo={repo_name}, pr={pr_number}, comments={len(comments) if comments else 0}")
    try:
        access_token = getenv("GITHUB_ACCESS_TOKEN")
        if not access_token:
            return "Error: GITHUB_ACCESS_TOKEN not found"
        
        g = Github(access_token)
        repo = g.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        head_repo = pr.head.repo
        commit = head_repo.get_commit(pr.head.sha)
        
        print(f"[DEBUG] About to create {len(comments)} inline comments")
        
        summary = "## 👋 PR Review Complete!\n\n"
        summary += "I've reviewed your changes and left some feedback inline. "
        summary += f"Found {len(comments)} items to discuss. "
        summary += "Check out the individual comments for details! ✨"
        pr.create_issue_comment(summary)
        
        comment_urls = []
        for comment_data in comments:
            comment = pr.create_comment(
                comment_data["body"],
                commit,
                comment_data["path"],
                comment_data["position"]
            )
            comment_urls.append(comment.html_url)
        
        return f"Posted review summary and {len(comment_urls)} inline comments"
    except Exception as e:
        return f"Error: {str(e)}"