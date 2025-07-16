import pytest
from unittest.mock import Mock, patch
import os
from typing import Any, Dict, List


def create_inline_pr_comments_logic(
    repo_name: str, pr_number: int, comments: List[Dict[str, Any]]
) -> str:
    """Logic for creating inline PR comments - extracted for testing."""
    try:
        access_token = os.getenv("GITHUB_ACCESS_TOKEN")
        if not access_token:
            return "Error: GITHUB_ACCESS_TOKEN not found"

        from github import Github

        g = Github(access_token)
        repo = g.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        head_repo = pr.head.repo
        commit = head_repo.get_commit(pr.head.sha)

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
                comment_data["position"],
            )
            comment_urls.append(comment.html_url)

        return f"Posted review summary and {len(comment_urls)} inline comments"
    except Exception as e:
        return f"Error: {str(e)}"


@pytest.fixture
def sample_comments():
    """Sample comment data for testing."""
    return [
        {
            "path": "src/main.py",
            "position": 10,
            "body": "Consider using a more descriptive variable name",
        },
        {
            "path": "tests/test_main.py",
            "position": 25,
            "body": "Add assertion message for better debugging",
        },
    ]


def test_create_inline_pr_comments_no_token():
    """Test error when GitHub token is not available."""
    with patch("os.getenv", return_value=None):
        result = create_inline_pr_comments_logic("owner/repo", 123, [])
        assert result == "Error: GITHUB_ACCESS_TOKEN not found"


def test_create_inline_pr_comments_success(sample_comments):
    """Test successful creation of inline PR comments."""
    # Create mock objects
    mock_comment = Mock()
    mock_comment.html_url = "https://github.com/owner/repo/pull/123#discussion_r456"

    mock_commit = Mock()

    mock_head_repo = Mock()
    mock_head_repo.get_commit.return_value = mock_commit

    mock_pr = Mock()
    mock_pr.head.sha = "abc123"
    mock_pr.head.repo = mock_head_repo
    mock_pr.create_comment.return_value = mock_comment
    mock_pr.create_issue_comment.return_value = Mock()

    mock_repo = Mock()
    mock_repo.get_pull.return_value = mock_pr

    mock_github = Mock()
    mock_github.get_repo.return_value = mock_repo

    with (
        patch("os.getenv", return_value="dummy_token"),
        patch("github.Github", return_value=mock_github),
    ):
        result = create_inline_pr_comments_logic("owner/repo", 123, sample_comments)

        assert "Posted review summary and 2 inline comments" in result
        mock_pr.create_issue_comment.assert_called_once()
        assert mock_pr.create_comment.call_count == 2


def test_create_inline_pr_comments_empty_list():
    """Test with empty comments list."""
    mock_commit = Mock()

    mock_head_repo = Mock()
    mock_head_repo.get_commit.return_value = mock_commit

    mock_pr = Mock()
    mock_pr.head.sha = "abc123"
    mock_pr.head.repo = mock_head_repo
    mock_pr.create_issue_comment.return_value = Mock()

    mock_repo = Mock()
    mock_repo.get_pull.return_value = mock_pr

    mock_github = Mock()
    mock_github.get_repo.return_value = mock_repo

    with (
        patch("os.getenv", return_value="dummy_token"),
        patch("github.Github", return_value=mock_github),
    ):
        result = create_inline_pr_comments_logic("owner/repo", 123, [])

        assert "Posted review summary and 0 inline comments" in result
        mock_pr.create_issue_comment.assert_called_once()
        mock_pr.create_comment.assert_not_called()


def test_create_inline_pr_comments_github_error():
    """Test error handling when GitHub API fails."""
    with (
        patch("os.getenv", return_value="dummy_token"),
        patch("github.Github", side_effect=Exception("GitHub API error")),
    ):
        result = create_inline_pr_comments_logic("owner/repo", 123, [])
        assert "Error: GitHub API error" in result


def test_create_inline_pr_comments_single_comment():
    """Test with single comment."""
    mock_comment = Mock()
    mock_comment.html_url = "https://github.com/owner/repo/pull/123#discussion_r456"

    mock_commit = Mock()

    mock_head_repo = Mock()
    mock_head_repo.get_commit.return_value = mock_commit

    mock_pr = Mock()
    mock_pr.head.sha = "abc123"
    mock_pr.head.repo = mock_head_repo
    mock_pr.create_comment.return_value = mock_comment
    mock_pr.create_issue_comment.return_value = Mock()

    mock_repo = Mock()
    mock_repo.get_pull.return_value = mock_pr

    mock_github = Mock()
    mock_github.get_repo.return_value = mock_repo

    single_comment = [
        {
            "path": "src/utils.py",
            "position": 5,
            "body": "This function could be optimized",
        }
    ]

    with (
        patch("os.getenv", return_value="dummy_token"),
        patch("github.Github", return_value=mock_github),
    ):
        result = create_inline_pr_comments_logic("owner/repo", 123, single_comment)

        assert "Posted review summary and 1 inline comments" in result
        mock_pr.create_comment.assert_called_once_with(
            "This function could be optimized", mock_commit, "src/utils.py", 5
        )


def test_create_inline_pr_comments_invalid_input():
    """Test with invalid input parameters."""
    with patch("os.getenv", return_value="dummy_token"):
        # None comments - this will cause TypeError when iterating
        result = create_inline_pr_comments_logic("owner/repo", 123, None)
        assert "Error:" in result


def test_create_inline_pr_comments_repo_not_found():
    """Test error when repository is not found."""
    mock_github = Mock()
    mock_github.get_repo.side_effect = Exception("Repository not found")

    with (
        patch("os.getenv", return_value="dummy_token"),
        patch("github.Github", return_value=mock_github),
    ):
        result = create_inline_pr_comments_logic("owner/nonexistent", 123, [])
        assert "Error: Repository not found" in result


def test_create_inline_pr_comments_pr_not_found():
    """Test error when pull request is not found."""
    mock_repo = Mock()
    mock_repo.get_pull.side_effect = Exception("Pull request not found")

    mock_github = Mock()
    mock_github.get_repo.return_value = mock_repo

    with (
        patch("os.getenv", return_value="dummy_token"),
        patch("github.Github", return_value=mock_github),
    ):
        result = create_inline_pr_comments_logic("owner/repo", 999, [])
        assert "Error: Pull request not found" in result


def test_create_inline_pr_comments_comment_creation_fails():
    """Test error when individual comment creation fails."""
    mock_commit = Mock()

    mock_head_repo = Mock()
    mock_head_repo.get_commit.return_value = mock_commit

    mock_pr = Mock()
    mock_pr.head.sha = "abc123"
    mock_pr.head.repo = mock_head_repo
    mock_pr.create_comment.side_effect = Exception("Comment creation failed")
    mock_pr.create_issue_comment.return_value = Mock()

    mock_repo = Mock()
    mock_repo.get_pull.return_value = mock_pr

    mock_github = Mock()
    mock_github.get_repo.return_value = mock_repo

    single_comment = [{"path": "src/test.py", "position": 1, "body": "Test comment"}]

    with (
        patch("os.getenv", return_value="dummy_token"),
        patch("github.Github", return_value=mock_github),
    ):
        result = create_inline_pr_comments_logic("owner/repo", 123, single_comment)
        assert "Error: Comment creation failed" in result
