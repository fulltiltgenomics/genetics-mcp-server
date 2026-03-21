"""Tests for the subagent system: skills, sandbox tools, and subagent service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genetics_mcp_server.skills.definitions import (
    SKILL_REGISTRY,
    get_skill,
    get_skill_descriptions,
    get_skill_instructions,
)
from genetics_mcp_server.skills.sandbox_tools import (
    _validate_path,
    execute_script,
    get_sandbox_tool_definitions,
    list_directory,
    read_file,
)
from genetics_mcp_server.subagent import SubagentResult, SubagentService


class TestSkillDefinitions:
    """Tests for skill definitions and loading."""

    def test_all_skills_have_instruction_files(self):
        """Every registered skill must have an instruction file that exists."""
        for skill in SKILL_REGISTRY.values():
            instructions = get_skill_instructions(skill)
            assert instructions, f"Skill '{skill.name}' has empty or missing instructions"

    def test_get_skill_returns_valid_skill(self):
        skill = get_skill("genetics_data_extraction")
        assert skill is not None
        assert skill.name == "genetics_data_extraction"

    def test_get_skill_returns_none_for_unknown(self):
        assert get_skill("nonexistent_skill") is None

    def test_get_skill_descriptions_format(self):
        desc = get_skill_descriptions()
        assert "genetics_data_extraction" in desc
        assert "literature_review" in desc
        assert "bigquery_analysis" in desc
        assert "data_analysis" in desc

    def test_skill_categories_are_valid(self):
        valid = {"general", "api", "bigquery"}
        for skill in SKILL_REGISTRY.values():
            assert skill.tool_categories.issubset(valid), (
                f"Skill '{skill.name}' has invalid categories: {skill.tool_categories - valid}"
            )

    def test_instruction_caching(self):
        """Loading same instruction twice returns cached result."""
        skill = SKILL_REGISTRY["literature_review"]
        first = get_skill_instructions(skill)
        second = get_skill_instructions(skill)
        assert first is second  # same object due to lru_cache


class TestSandboxPathValidation:
    """Tests for path security in sandbox tools."""

    def test_valid_path_under_allowed(self, tmp_path):
        allowed = [str(tmp_path)]
        test_file = tmp_path / "test.txt"
        test_file.touch()
        result = _validate_path(str(test_file), allowed)
        assert result == test_file.resolve()

    def test_path_traversal_blocked(self, tmp_path):
        allowed = [str(tmp_path)]
        with pytest.raises(ValueError, match="outside allowed"):
            _validate_path(str(tmp_path / ".." / "etc" / "passwd"), allowed)

    def test_no_allowed_paths_raises(self):
        with pytest.raises(ValueError, match="No allowed paths"):
            _validate_path("/some/path", [])

    def test_symlink_traversal_blocked(self, tmp_path):
        """Symlinks that escape allowed paths are blocked."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("secret")
        link = allowed_dir / "link.txt"
        link.symlink_to(secret)
        # the resolved path of the link is outside allowed_dir
        with pytest.raises(ValueError, match="outside allowed"):
            _validate_path(str(link), [str(allowed_dir)])


class TestSandboxFileOps:
    """Tests for sandbox file read and directory listing."""

    @pytest.mark.asyncio
    async def test_read_file_success(self, tmp_path):
        test_file = tmp_path / "data.txt"
        test_file.write_text("hello world")
        result = await read_file(str(test_file), [str(tmp_path)])
        assert result["success"] is True
        assert result["content"] == "hello world"
        assert result["truncated"] is False

    @pytest.mark.asyncio
    async def test_read_file_outside_allowed(self, tmp_path):
        result = await read_file("/etc/passwd", [str(tmp_path)])
        assert result["success"] is False
        assert "outside allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, tmp_path):
        result = await read_file(str(tmp_path / "missing.txt"), [str(tmp_path)])
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_list_directory_success(self, tmp_path):
        (tmp_path / "file1.txt").touch()
        (tmp_path / "subdir").mkdir()
        result = await list_directory(str(tmp_path), [str(tmp_path)])
        assert result["success"] is True
        names = [e["name"] for e in result["entries"]]
        assert "file1.txt" in names
        assert "subdir" in names

    @pytest.mark.asyncio
    async def test_list_directory_outside_allowed(self, tmp_path):
        result = await list_directory("/etc", [str(tmp_path)])
        assert result["success"] is False


class TestSandboxScriptExecution:
    """Tests for sandbox script execution."""

    @pytest.mark.asyncio
    async def test_execute_python_script(self, tmp_path):
        result = await execute_script(
            interpreter="python3",
            script="print('hello from python')",
            working_dir=str(tmp_path),
            allowed_paths=[str(tmp_path)],
        )
        assert result["success"] is True
        assert "hello from python" in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_bash_script(self, tmp_path):
        result = await execute_script(
            interpreter="bash",
            script="echo 'hello from bash'",
            working_dir=str(tmp_path),
            allowed_paths=[str(tmp_path)],
        )
        assert result["success"] is True
        assert "hello from bash" in result["stdout"]

    @pytest.mark.asyncio
    async def test_disallowed_interpreter(self, tmp_path):
        result = await execute_script(
            interpreter="perl",
            script="print 'hello'",
            working_dir=str(tmp_path),
            allowed_paths=[str(tmp_path)],
        )
        assert result["success"] is False
        assert "not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_script_timeout(self, tmp_path):
        result = await execute_script(
            interpreter="python3",
            script="import time; time.sleep(10)",
            working_dir=str(tmp_path),
            allowed_paths=[str(tmp_path)],
            timeout=1,
        )
        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_script_working_dir_restricted(self, tmp_path):
        result = await execute_script(
            interpreter="python3",
            script="print('hi')",
            working_dir="/etc",
            allowed_paths=[str(tmp_path)],
        )
        assert result["success"] is False
        assert "outside allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_sensitive_env_stripped(self, tmp_path):
        """API keys should not leak into script environment."""
        result = await execute_script(
            interpreter="python3",
            script="import os; print(os.environ.get('ANTHROPIC_API_KEY', 'NOT_SET'))",
            working_dir=str(tmp_path),
            allowed_paths=[str(tmp_path)],
        )
        assert result["success"] is True
        assert "NOT_SET" in result["stdout"]

    @pytest.mark.asyncio
    async def test_script_failure_returns_stderr(self, tmp_path):
        result = await execute_script(
            interpreter="python3",
            script="raise ValueError('test error')",
            working_dir=str(tmp_path),
            allowed_paths=[str(tmp_path)],
        )
        assert result["success"] is False
        assert result["return_code"] != 0
        assert "test error" in result["stderr"]


class TestSandboxToolDefinitions:
    """Tests for sandbox tool definition generation."""

    def test_no_tools_when_disabled(self):
        tools = get_sandbox_tool_definitions(False, False)
        assert tools == []

    def test_file_tools_when_read_enabled(self):
        tools = get_sandbox_tool_definitions(True, False)
        names = [t["name"] for t in tools]
        assert "read_file" in names
        assert "list_directory" in names
        assert "execute_script" not in names

    def test_all_tools_when_both_enabled(self):
        tools = get_sandbox_tool_definitions(True, True)
        names = [t["name"] for t in tools]
        assert "read_file" in names
        assert "list_directory" in names
        assert "execute_script" in names


class TestSubagentResult:
    """Tests for SubagentResult dataclass."""

    def test_default_values(self):
        result = SubagentResult(skill_name="test", query="q", output="o")
        assert result.success is True
        assert result.error is None
        assert result.tools_used == []
        assert result.iterations == 0


class TestSubagentService:
    """Tests for SubagentService with mocked Claude API."""

    def _make_mock_message(self, text="result text", tool_uses=None):
        """Create a mock Anthropic message response."""
        content = []
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        content.append(text_block)

        if tool_uses:
            for tu in tool_uses:
                block = MagicMock()
                block.type = "tool_use"
                block.id = tu["id"]
                block.name = tu["name"]
                block.input = tu["input"]
                block.model_dump.return_value = {
                    "type": "tool_use",
                    "id": tu["id"],
                    "name": tu["name"],
                    "input": tu["input"],
                }
                content.append(block)

        msg = MagicMock()
        msg.content = content
        msg.stop_reason = "end_turn" if not tool_uses else "tool_use"
        return msg

    @pytest.mark.asyncio
    async def test_run_subagent_simple(self):
        """Test a subagent that returns text without tool calls."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=self._make_mock_message("Analysis complete.")
        )
        mock_executor = MagicMock()

        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("literature_review")

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            settings = MagicMock()
            settings.subagent_model = ""
            settings.fast_model = "claude-haiku-4-5"
            settings.temperature = 0.3
            settings.mcp_max_result_size = 50000
            settings.subagent_timeout = 120
            settings.subagent_script_timeout = 30
            settings.enable_subagents = True
            settings.enable_script_execution = False
            settings.disabled_tools = set()
            settings.subagent_allowed_paths_list = []
            mock_settings.return_value = settings

            result = await service._run_subagent(skill, "Find papers about PCSK9")

        assert result.success is True
        assert result.output == "Analysis complete."
        assert result.skill_name == "literature_review"

    @pytest.mark.asyncio
    async def test_run_subagents_parallel(self):
        """Test running multiple subagents in parallel."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=self._make_mock_message("Done.")
        )
        mock_executor = MagicMock()

        service = SubagentService(mock_client, mock_executor)

        tasks = [
            {"skill": "genetics_data_extraction", "query": "Get GWAS data for PCSK9"},
            {"skill": "literature_review", "query": "Find papers about PCSK9"},
        ]

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            settings = MagicMock()
            settings.subagent_model = ""
            settings.fast_model = "claude-haiku-4-5"
            settings.temperature = 0.3
            settings.mcp_max_result_size = 50000
            settings.subagent_timeout = 120
            settings.subagent_script_timeout = 30
            settings.enable_subagents = True
            settings.enable_script_execution = False
            settings.disabled_tools = set()
            settings.subagent_allowed_paths_list = []
            mock_settings.return_value = settings

            result = await service.run_subagents(tasks)

        assert result["success"] is True
        assert len(result["results"]) == 2
        assert all(r["success"] for r in result["results"])

    @pytest.mark.asyncio
    async def test_unknown_skill_returns_error(self):
        mock_client = MagicMock()
        mock_executor = MagicMock()
        service = SubagentService(mock_client, mock_executor)

        result = await service.run_subagents([{"skill": "nonexistent", "query": "test"}])
        assert result["success"] is False
        assert "Unknown skill" in result["error"]
