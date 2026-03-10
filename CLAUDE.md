====


QUALITY CODING RULES


# Code changes

1. If you find errors or suggestions in code which are not DIRECTLY related to user's current request, never change it without asking first.
2. Before suggesting changes to files, always assume user might have changed the file since your last read and consider reading the file again.


# Security

1. Never commit sensitive files
2. Use environment variables for API keys and credentials
3. Keep API keys and credentials out of logs and output


# Project Specifications

1. Project documentation is maintained in files in `docs/` folder.
2. `docs/project-spec.md` is an overview of project purpose, structure and logic.
3. Create other files under `docs/` if necessary.
4. Maintain `docs/project-spec.md` and any other generated files to be up to date with the project.
5. Reread `docs/project-spec.md` often and whenever you need to refresh your context with what the project is about and implementation logic.
6. This should often be your first step in understanding a task.


# Software Development Behavior Guidelines

1. Don't guess and do things which you are not certain about. Ask the user instead.
2. Don't add or modify code unrelated to the specific request and context at the moment.
3. Only use git when asked, and when using git, only stage changes and propose a commit message. Let the user review the changes and commit them.
4. **Always** prior to finishing a task and considering it completed, revise all the changes and update Project Specification files.
5. **Always** prior to finishing a task and considering it completed, make sure all tests run successfully.
6. When trying to fix any bug or compiler error **ALWAYS** think carefully and analyze in detail what happened and WHY? Explain and confirm with user.


# Code Conventions

1. Project structure contains `docs/`, `src/` and `tests/` folders at the root
2. Code should be self-descriptive
   - Only add comments for tricky or complex parts of the code (explaining WHY something is done)
   - NO redundant and trivial comments that simply restate what the code does
3. This project uses async/await throughout for I/O operations
   - All HTTP calls use httpx.AsyncClient
   - Tool executor methods are async
   - MCP and LLM service handlers are async
4. Private fields and methods should be prefixed with underscore
5. Code should pass linting at all times (`ruff check src/`)


# Project-specific conventions

1. Tool definitions live in `src/genetics_mcp_server/tools/definitions.py`
   - Both MCP server and LLM service use these definitions
   - Keep them in sync when adding or modifying tools
2. Tool execution logic lives in `src/genetics_mcp_server/tools/executor.py`
   - Each tool is an async method on the ToolExecutor class
   - Return `{"success": True, ...}` on success, `{"success": False, "error": "..."}` on failure
3. Configuration uses environment variables loaded via python-dotenv
   - All settings defined in `src/genetics_mcp_server/config/settings.py`
   - Document new variables in `.env.example`
4. Tests use pytest with pytest-asyncio
   - Run with `pytest` or `pytest --cov` for coverage


====

**Don't forget any of the 'QUALITY CODING RULES' above!!!**
