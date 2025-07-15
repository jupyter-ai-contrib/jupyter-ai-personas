import functools
import io
import traceback
from pathlib import Path
from typing import Any, List, Optional, Dict, Union
from contextlib import redirect_stdout, redirect_stderr, contextmanager

from agno.tools import Toolkit
from agno.utils.log import log_debug, log_info, logger


@functools.lru_cache(maxsize=None)
def warn() -> None:
    logger.warning("ImprovedPythonTools can run arbitrary code, please provide human supervision.")


@contextmanager
def change_dir(path):
    """Context manager to temporarily change working directory."""
    import os
    original = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original)


class ImprovedPythonTools(Toolkit):
    def __init__(
        self,
        session_dir: Optional[Path] = None,
        run_code: bool = True,
        save_essential_files: bool = True,
        read_files: bool = True,
        list_files: bool = True,
        safe_globals: Optional[dict] = None,
        safe_locals: Optional[dict] = None,
        **kwargs,
    ):
        """
        Improved Python tools that execute code in-memory and only save essential files.
        
        Args:
            session_dir: Directory for saving essential files (CSV, images, etc.)
            run_code: Enable code execution
            save_essential_files: Enable saving important files (data, plots)
            read_files: Enable file reading
            list_files: Enable directory listing
            safe_globals: Global scope for code execution
            safe_locals: Local scope for code execution
        """
        # CRITICAL FIX: Don't default to Path.cwd() - use the provided session_dir
        if session_dir is not None:
            self.session_dir: Path = Path(session_dir)
        else:
            # Only use current directory as absolute last resort
            self.session_dir: Path = Path.cwd()
            logger.warning(f"No session_dir provided to ImprovedPythonTools, using: {self.session_dir}")

        # Ensure the directory exists
        self.session_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ImprovedPythonTools using session directory: {self.session_dir.absolute()}")

        # Execution environment
        self.safe_globals: dict = safe_globals or {
            '__builtins__': __builtins__,
            'print': print,
        }
        self.safe_locals: dict = safe_locals or {}

        # Persistent variables across executions
        self.execution_context: Dict[str, Any] = {}

        tools: List[Any] = []
        if run_code:
            tools.append(self.run_python_code)
        if save_essential_files:
            tools.append(self.save_essential_file)
        if read_files:
            tools.append(self.read_file)
        if list_files:
            tools.append(self.list_files)

        super().__init__(name="improved_python_tools", tools=tools, **kwargs)

    def run_python_code(self, code: str, description: str = "") -> str:
        """
        Execute Python code in-memory without saving to files.
        Maintains execution context between calls.
        
        Args:
            code: Python code to execute
            description: Optional description of what the code does
            
        Returns:
            Execution result, output, or error message
        """
        try:
            warn()

            if description:
                log_info(f"Executing: {description}")

            log_debug(f"Running code:\n{code}\n")

            # Capture stdout and stderr
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            # CRITICAL FIX: Inject session directory into execution context
            execution_globals = {
                **self.safe_globals, 
                **self.execution_context,
                # Force the session directory to be available
                'SESSION_DIR': str(self.session_dir),
                'session_dir': str(self.session_dir),
            }
            execution_locals = {**self.safe_locals}

            # CRITICAL FIX: Change working directory before execution
            log_info(f"Changed working directory to: {self.session_dir}")
            
            with change_dir(self.session_dir):
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    exec(code, execution_globals, execution_locals)

            # Update execution context with new variables
            self.execution_context.update(execution_locals)

            # Get captured output
            stdout_output = stdout_buffer.getvalue()
            stderr_output = stderr_buffer.getvalue()

            result_parts = []
            if stdout_output.strip():
                result_parts.append(f"Output:\n{stdout_output.strip()}")
            if stderr_output.strip():
                result_parts.append(f"Stderr:\n{stderr_output.strip()}")

            if not result_parts:
                result_parts.append("Code executed successfully (no output)")

            return "\n\n".join(result_parts)

        except Exception as e:
            error_msg = f"Error executing code: {str(e)}\n"
            error_msg += f"Traceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            return error_msg

    def save_essential_file(self, file_path: Union[str, Path], content: Any, file_type: str = "auto") -> str:
        """
        Save only essential files (data, plots, results) to the session directory.
        
        Args:
            file_path: Relative path within session directory (str or Path)
            content: Content to save (DataFrame, plot figure, string, etc.)
            file_type: Type of file ('csv', 'png', 'txt', 'auto')
            
        Returns:
            Success or error message
        """
        try:
            # CRITICAL FIX: Always use session directory for file operations
            full_path = self.session_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            log_info(f"Saving file to session directory: {full_path}")

            # Auto-detect file type from extension if not specified
            if file_type == "auto":
                file_type = full_path.suffix.lower().lstrip('.')

            if file_type in ['csv']:
                # Handle pandas DataFrame
                if hasattr(content, 'to_csv'):
                    content.to_csv(full_path, index=False)
                else:
                    # Handle string CSV content
                    full_path.write_text(str(content), encoding='utf-8')

            elif file_type in ['png', 'jpg', 'jpeg', 'svg', 'pdf']:
                # Handle matplotlib figure
                if hasattr(content, 'savefig'):
                    content.savefig(full_path, dpi=300, bbox_inches='tight')
                else:
                    # Handle raw image data
                    with open(full_path, 'wb') as f:
                        f.write(content)

            elif file_type in ['txt', 'md', 'json', 'yaml', 'yml']:
                # Handle text content
                full_path.write_text(str(content), encoding='utf-8')

            else:
                # Generic binary write
                if isinstance(content, (str, bytes)):
                    mode = 'w' if isinstance(content, str) else 'wb'
                    with open(full_path, mode) as f:
                        f.write(content)
                else:
                    return f"Unsupported content type for file: {file_path}"

            log_info(f"Successfully saved essential file: {full_path}")
            return f"Successfully saved: {file_path} in {self.session_dir}"

        except Exception as e:
            error_msg = f"Error saving file {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def get_variable_str_representation(self, variable_name: str) -> str:
        """
        Get the value of a variable from the execution context.
        
        Args:
            variable_name: Name of the variable to retrieve
            
        Returns:
            String representation of the variable value
        """
        try:
            if variable_name in self.execution_context:
                value = self.execution_context[variable_name]
                return str(value)
            else:
                return f"Variable '{variable_name}' not found in execution context"
        except Exception as e:
            return f"Error retrieving variable {variable_name}: {str(e)}"

    def list_variables(self) -> str:
        """
        List all variables in the current execution context.
        
        Returns:
            String listing all available variables and their types
        """
        try:
            if not self.execution_context:
                return "No variables in execution context"

            var_list = []
            for name, value in self.execution_context.items():
                if not name.startswith('_'):  # Skip private variables
                    # Get more detailed type information
                    var_type = str(type(value))
                    # Clean up the type string for readability
                    if var_type.startswith("<class '") and var_type.endswith("'>"):
                        var_type = var_type[8:-2]
                    var_list.append(f"{name}: {var_type}")

            return "Available variables:\n" + "\n".join(var_list)

        except Exception as e:
            return f"Error listing variables: {str(e)}"

    def read_file(self, file_name: str) -> str:
        """
        Read contents of a file from the session directory.
        
        Args:
            file_name: Name of the file to read
            
        Returns:
            File contents or error message
        """
        log_info(f"Reading file: {file_name} from {self.session_dir}")
        file_path = self.session_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_name} in {self.session_dir}")
        try:
            contents = file_path.read_text(encoding="utf-8")
            return contents
        except Exception as e:
            error_msg = f"Error reading file {file_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise IOError(error_msg) from e

    def list_files(self, file_pattern: str = "*") -> str:
        """
        List files in the session directory.
        
        Args:
            file_pattern: Glob pattern to filter files (default: all files)
            
        Returns:
            Comma-separated list of files matching the pattern
        """
        try:
            log_info(f"Listing files in: {self.session_dir}")
            files = list(self.session_dir.glob(file_pattern))
            file_names = [f.name for f in files if f.is_file()]

            if not file_names:
                return f"No files found matching pattern: {file_pattern} in {self.session_dir}"

            return f"Files in session directory ({self.session_dir}): {', '.join(sorted(file_names))}"

        except Exception as e:
            error_msg = f"Error listing files: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def clear_context(self) -> str:
        """
        Clear the execution context (remove all variables).
        
        Returns:
            Confirmation message
        """
        self.execution_context.clear()
        return "Execution context cleared"

    def execute_visualization_code(self, code: str, plot_filename: str) -> str:
        """
        Execute visualization code and automatically save the plot.
        
        Args:
            code: Python code that creates a matplotlib plot
            plot_filename: Filename to save the plot (with extension)
            
        Returns:
            Success message or error
        """
        try:
            # CRITICAL FIX: Ensure plot is saved in session directory
            plot_path = self.session_dir / plot_filename

            # Add plot saving to the code
            enhanced_code = f"""
                import matplotlib.pyplot as plt
                {code}
                # Auto-save the plot to session directory
                plt.savefig(r'{plot_path}', dpi=300, bbox_inches='tight')
                plt.close()  # Clean up memory
                print(f'Plot saved to: {plot_path}')
                """

            result = self.run_python_code(enhanced_code, f"Creating visualization: {plot_filename}")

            if "Error" not in result:
                return f"Visualization saved: {plot_filename} in {self.session_dir}\n{result}"
            else:
                return result

        except Exception as e:
            return f"Error creating visualization: {str(e)}"
