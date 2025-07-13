from typing import Any, Dict, List
from github import Github
from os import getenv
from agno.tools import tool
import logging

logger = logging.getLogger(__name__)


@tool
def create_inline_pr_comments(
    repo_name: str, pr_number: int, comments: List[Dict[str, Any]]
) -> str:
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
    logger.debug(
        f"create_inline_pr_comments called with repo={repo_name}, pr={pr_number}, comments={len(comments) if comments else 0}"
    )
    try:
        access_token = getenv("GITHUB_ACCESS_TOKEN")
        if not access_token:
            return "Error: GITHUB_ACCESS_TOKEN not found"

        g = Github(access_token)
        repo = g.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        head_repo = pr.head.repo
        commit = head_repo.get_commit(pr.head.sha)

        logger.debug(f"About to create {len(comments)} inline comments")

        #  high-level summary
        summary = f"## 🔍 PR Review Summary\n\n"
        summary += f"Found {len(comments)} issues that need attention. "
        summary += "Please check the inline comments below for specific details.\n\n"
        summary += "**Key Areas:**\n"
        summary += "- Code quality and best practices\n"
        summary += "- Security considerations\n"
        summary += "- Documentation completeness\n\n"
        summary += "_Review completed by AI Assistant_"

        try:
            pr.create_review(body=summary, event="COMMENT")
            logger.debug("Created summary review")
        except Exception as e:
            logger.warning(f"Summary creation failed: {str(e)}")

        # map line numbers to diff positions
        pr_files = {f.filename: f for f in pr.get_files()}

        comment_urls = []
        errors = []

        for i, comment_data in enumerate(comments):
            logger.debug(f"Processing comment {i + 1}")
            try:
                if not all(key in comment_data for key in ["path", "position", "body"]):
                    logger.warning(f"Comment {i + 1}: Missing required fields")
                    errors.append(f"Comment {i + 1}: Missing required fields")
                    continue

                file_path = comment_data["path"]
                line_number = comment_data["position"]
                logger.debug(f"Comment {i + 1}: Targeting {file_path}:{line_number}")

                if file_path not in pr_files:
                    logger.warning(
                        f"Comment {i + 1}: File {file_path} not found in PR files"
                    )
                    errors.append(f"Comment {i + 1}: File {file_path} not in PR")
                    continue

                # Try direct line number first
                try:
                    logger.debug(
                        f"Comment {i + 1}: Attempting inline comment at line {line_number}"
                    )
                    comment = pr.create_comment(
                        comment_data["body"],
                        commit,
                        file_path,
                        line_number,
                    )
                    comment_urls.append(comment.html_url)
                    logger.info(f"Comment {i + 1}: Successfully created inline comment")
                    continue
                except Exception as inline_error:
                    logger.warning(
                        f"Comment {i + 1}: Inline failed: {str(inline_error)}"
                    )

                # If inline fails, just log and skip
                error_msg = f"Comment {i + 1} failed: {str(inline_error)}"
                errors.append(error_msg)
                logger.error(error_msg)

            except Exception as e:
                error_msg = f"Comment {i + 1} failed: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        success_count = len(comment_urls)
        error_count = len(errors)

        result = f"Posted {success_count} comments"
        if errors:
            result += f", {error_count} failed: {'; '.join(errors[:3])}"

        return result
    except Exception as e:
        return f"Error: {str(e)}"
