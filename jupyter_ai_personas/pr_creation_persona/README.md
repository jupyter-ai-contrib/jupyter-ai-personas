# PR Creation Persona

A specialized AI assistant for analyzing GitHub issues and implementing code fixes with automated git operations.

## Overview

The PR Creation Persona coordinates a team of specialized agents to:
1. **Analyze Issues** - Parse requirements and understand scope
2. **Design Architecture** - Plan minimal solution architecture  
3. **Implement Code** - Write focused code that addresses the issue
4. **Manage Git Operations** - Handle cloning, branching, committing, and pushing

## Key Features

### Clear Task Separation
- **Issue Analysis Agent**: Parses requirements and analyzes repository context
- **Architecture Designer**: Plans implementation strategy and file changes
- **Code Implementer**: Writes minimal, focused code following existing patterns
- **Git Manager**: Handles all git operations including branch creation and pushing

### Repository Context Awareness
- Uses knowledge graph analysis to understand codebase structure
- Identifies existing patterns and conventions
- Analyzes dependencies and relationships
- Finds similar implementations for reference

### Minimal Code Implementation
- Writes only the absolute minimum code needed
- Follows existing code patterns and style
- Focuses specifically on issue requirements
- Avoids verbose or unnecessary implementations

### Complete Git Workflow
- Uses Agno's ShellTools for git operations
- Clones main branch automatically
- Creates descriptive feature branches
- Commits changes with clear messages
- Pushes to remote branch (does NOT create PR)

## Usage

Provide an issue description along with a GitHub repository URL:

```
Analyze this issue and implement a fix:

Repository: https://github.com/user/repo
Issue: Add validation to user input in the login form to prevent SQL injection

The login form currently accepts any input without validation...
```

### Using with Local Repository

If you're already working on a repository locally, you can set the `LOCAL_REPO_PATH` environment variable to use your existing clone:

```bash
# Set the environment variable to your local repository path
export LOCAL_REPO_PATH=/path/to/your/local/repo

# Then use the persona as normal - it will use your local repository
# instead of cloning a new one
```

If the specified `LOCAL_REPO_PATH` doesn't exist but the environment variable is set, the persona will automatically clone the repository to that location when processing an issue. This allows you to specify where you want the repository to be cloned without having to create the directory structure yourself.

## Workflow

### Phase 1: Issue Analysis
- Extracts issue requirements and acceptance criteria
- Uses KG queries to understand repository structure
- Identifies affected components and files
- Assesses scope and complexity

### Phase 2: Architecture Design
- Designs minimal solution architecture
- Plans file structure and organization
- Defines implementation strategy
- Creates detailed file-by-file changes plan

### Phase 3: Code Implementation
- Uses Agno's ShellTools and FileTools
- Sets up repository and creates feature branch
- Implements code following the architecture plan
- Writes minimal, focused code addressing the issue
- Maintains consistency with existing patterns

### Phase 4: Git Operations
- Uses standard git commands via ShellTools
- Commits changes with descriptive messages
- Pushes feature branch to remote repository
- Provides branch information for manual PR creation
- Does NOT automatically create pull requests

## Requirements

- GitHub personal access token in `GITHUB_ACCESS_TOKEN` environment variable
- Neo4j database running on `neo4j://127.0.0.1:7687`
- AWS credentials configured for Bedrock access

## Environment Variables

- `GITHUB_ACCESS_TOKEN`: Required. GitHub personal access token for repository access.
- `LOCAL_REPO_PATH`: Optional. Path to an existing local repository clone. When provided, the persona will use this local repository instead of cloning a new one, which is ideal for developers already working on the repository.

## Output

The persona provides:
- Detailed issue analysis and requirements
- Solution architecture and implementation plan
- Code implementation with explanations
- Git operations summary with branch information
- Instructions for manual PR creation

## Integration

The persona integrates with:
- **Knowledge Graph**: For codebase analysis and context
- **GitHub API**: For repository access and metadata
- **Agno ShellTools**: For git operations and command execution
- **Agno FileTools**: For code creation and modification

## Best Practices

- Always analyzes repository context before implementation
- Maintains clear separation between analysis and implementation
- Writes minimal, focused code addressing specific issues
- Follows existing code patterns and conventions
- Provides complete git workflow without creating PRs
- Ensures proper error handling and validation