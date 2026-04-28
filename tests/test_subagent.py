"""Tests for the subagent system: skills, sandbox tools, and subagent service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genetics_mcp_server.llm_service import _process_download_hints
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
from genetics_mcp_server.subagent import SubagentResult, SubagentService, _format_tool_params


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
        assert result.subagent_id == ""

    def test_subagent_id_set(self):
        result = SubagentResult(skill_name="test", query="q", output="o", subagent_id="sa-3")
        assert result.subagent_id == "sa-3"


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


class TestOrchestrationCategoryExclusion:
    """Tests that orchestration tools (launch_subagents) are excluded from subagent tool sets."""

    def test_launch_subagents_excluded_from_subagent_tools(self):
        """Subagents must not be able to recursively launch subagents."""
        mock_client = MagicMock()
        mock_executor = MagicMock()
        service = SubagentService(mock_client, mock_executor)

        skill = get_skill("genetics_data_extraction")
        assert skill is not None

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            settings = MagicMock()
            settings.disabled_tools = set()
            settings.enable_subagents = True
            settings.enable_script_execution = False
            settings.subagent_allowed_paths_list = []
            mock_settings.return_value = settings

            tools = service._get_tool_definitions(skill)

        tool_names = [t["name"] for t in tools]
        assert "launch_subagents" not in tool_names

    def test_launch_subagents_excluded_even_when_in_profile(self):
        """Even for api/bigquery profiles that include orchestration, launch_subagents is disabled."""
        mock_client = MagicMock()
        mock_executor = MagicMock()
        service = SubagentService(mock_client, mock_executor)

        for skill_name in SKILL_REGISTRY:
            skill = get_skill(skill_name)
            with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
                settings = MagicMock()
                settings.disabled_tools = set()
                settings.enable_subagents = True
                settings.enable_script_execution = False
                settings.subagent_allowed_paths_list = []
                mock_settings.return_value = settings

                tools = service._get_tool_definitions(skill)

            tool_names = [t["name"] for t in tools]
            assert "launch_subagents" not in tool_names, (
                f"launch_subagents should be excluded for skill '{skill_name}'"
            )


class TestDownloadHintProcessing:
    """Tests that _execute_subagent_tool applies _process_download_hints on local tool results."""

    @pytest.mark.asyncio
    async def test_download_hints_called_on_local_tool_result(self):
        """Local tool results should be processed through _process_download_hints."""
        mock_client = MagicMock()
        mock_executor = MagicMock()
        raw_result = {"success": True, "data": "test"}
        mock_executor.some_tool = AsyncMock(return_value=raw_result)

        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("genetics_data_extraction")

        with (
            patch("genetics_mcp_server.subagent.get_settings") as mock_settings,
            patch("genetics_mcp_server.subagent.is_external_tool", return_value=False),
            patch(
                "genetics_mcp_server.llm_service._process_download_hints",
                wraps=_process_download_hints,
            ) as mock_hints,
        ):
            settings = MagicMock()
            settings.subagent_allowed_paths_list = []
            mock_settings.return_value = settings

            result = await service._execute_subagent_tool("some_tool", {}, skill)

        mock_hints.assert_called_once_with(raw_result)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_download_url_hint_converted(self):
        """_download_url in a result should become an INCLUDE_IN_RESPONSE link."""
        mock_client = MagicMock()
        mock_executor = MagicMock()
        raw_result = {
            "success": True,
            "_download_url": "https://example.com/api/download?id=123",
            "count": 5,
        }
        mock_executor.my_tool = AsyncMock(return_value=raw_result)

        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("genetics_data_extraction")

        with (
            patch("genetics_mcp_server.subagent.get_settings") as mock_settings,
            patch("genetics_mcp_server.subagent.is_external_tool", return_value=False),
        ):
            settings = MagicMock()
            settings.subagent_allowed_paths_list = []
            mock_settings.return_value = settings

            result = await service._execute_subagent_tool("my_tool", {}, skill)

        assert "INCLUDE_IN_RESPONSE" in result
        assert "Download" in result["INCLUDE_IN_RESPONSE"]
        assert "_download_url" not in result

    @pytest.mark.asyncio
    async def test_no_download_hints_on_failure(self):
        """Failed results should pass through without download hint processing."""
        mock_client = MagicMock()
        mock_executor = MagicMock()
        raw_result = {"success": False, "error": "something broke"}
        mock_executor.fail_tool = AsyncMock(return_value=raw_result)

        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("genetics_data_extraction")

        with (
            patch("genetics_mcp_server.subagent.get_settings") as mock_settings,
            patch("genetics_mcp_server.subagent.is_external_tool", return_value=False),
        ):
            settings = MagicMock()
            settings.subagent_allowed_paths_list = []
            mock_settings.return_value = settings

            result = await service._execute_subagent_tool("fail_tool", {}, skill)

        assert result == raw_result
        assert "INCLUDE_IN_RESPONSE" not in result


class TestTokenAccumulation:
    """Tests that SubagentResult accumulates input_tokens/output_tokens across iterations."""

    def _make_settings_mock(self):
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
        return settings

    def _make_message(self, text="done", tool_uses=None, input_tokens=100, output_tokens=50):
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
        msg.usage = MagicMock()
        msg.usage.input_tokens = input_tokens
        msg.usage.output_tokens = output_tokens
        return msg

    @pytest.mark.asyncio
    async def test_single_iteration_tokens(self):
        """Single API call should record its token usage."""
        msg = self._make_message(input_tokens=150, output_tokens=75)
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=msg)
        mock_executor = MagicMock()

        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("literature_review")

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            mock_settings.return_value = self._make_settings_mock()
            result = await service._run_subagent(skill, "test query")

        assert result.input_tokens == 150
        assert result.output_tokens == 75

    @pytest.mark.asyncio
    async def test_multi_iteration_token_accumulation(self):
        """Tokens should accumulate across multiple agentic loop iterations."""
        tool_msg = self._make_message(
            text="thinking",
            tool_uses=[{"id": "t1", "name": "search_variants", "input": {"query": "test"}}],
            input_tokens=200,
            output_tokens=100,
        )
        final_msg = self._make_message(text="final answer", input_tokens=300, output_tokens=150)

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=[tool_msg, final_msg])
        mock_executor = MagicMock()
        mock_executor.search_variants = AsyncMock(return_value={"success": True, "data": []})

        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("genetics_data_extraction")

        with (
            patch("genetics_mcp_server.subagent.get_settings") as mock_settings,
            patch("genetics_mcp_server.subagent.is_external_tool", return_value=False),
        ):
            mock_settings.return_value = self._make_settings_mock()
            result = await service._run_subagent(skill, "test query")

        assert result.input_tokens == 500  # 200 + 300
        assert result.output_tokens == 250  # 100 + 150
        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_tokens_in_parallel_results(self):
        """Token counts should appear in run_subagents output dicts."""
        msg = self._make_message(input_tokens=100, output_tokens=50)
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=msg)
        mock_executor = MagicMock()

        service = SubagentService(mock_client, mock_executor)
        tasks = [{"skill": "literature_review", "query": "test"}]

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            mock_settings.return_value = self._make_settings_mock()
            result = await service.run_subagents(tasks)

        assert result["success"] is True
        r = result["results"][0]
        assert r["input_tokens"] == 100
        assert r["output_tokens"] == 50


class TestProgressCallback:
    """Tests that _run_subagent invokes the progress_callback at key lifecycle points."""

    def _make_settings_mock(self):
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
        return settings

    def _make_message(self, text="done", tool_uses=None):
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
        msg.usage = MagicMock()
        msg.usage.input_tokens = 10
        msg.usage.output_tokens = 5
        return msg

    @pytest.mark.asyncio
    async def test_callback_on_start_and_completion(self):
        """Progress callback fires at start and completion with subagent ID."""
        msg = self._make_message("result")
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=msg)
        mock_executor = MagicMock()
        callback = MagicMock()

        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("literature_review")

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            mock_settings.return_value = self._make_settings_mock()
            await service._run_subagent(skill, "test", progress_callback=callback, subagent_id="sa-1")

        calls = [c.args[0] for c in callback.call_args_list]
        assert any("[sa-1]" in c and "started" in c for c in calls)
        assert any("[sa-1]" in c and "completed" in c for c in calls)

    @pytest.mark.asyncio
    async def test_callback_on_tool_call(self):
        """Progress callback fires with tool name, params, and subagent ID."""
        tool_msg = self._make_message(
            text="",
            tool_uses=[{"id": "t1", "name": "search_variants", "input": {"query": "BRCA1"}}],
        )
        final_msg = self._make_message("done")

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=[tool_msg, final_msg])
        mock_executor = MagicMock()
        mock_executor.search_variants = AsyncMock(return_value={"success": True})
        callback = MagicMock()

        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("genetics_data_extraction")

        with (
            patch("genetics_mcp_server.subagent.get_settings") as mock_settings,
            patch("genetics_mcp_server.subagent.is_external_tool", return_value=False),
        ):
            mock_settings.return_value = self._make_settings_mock()
            await service._run_subagent(skill, "test", progress_callback=callback, subagent_id="sa-2")

        calls = [c.args[0] for c in callback.call_args_list]
        assert any("[sa-2]" in c and "calling search_variants" in c for c in calls)
        # params should appear in the tool call message
        tool_call_msg = next(c for c in calls if "calling search_variants" in c)
        assert "query='BRCA1'" in tool_call_msg

    @pytest.mark.asyncio
    async def test_callback_on_failure(self):
        """Progress callback fires with failure message including subagent ID."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))
        mock_executor = MagicMock()
        callback = MagicMock()

        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("literature_review")

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            mock_settings.return_value = self._make_settings_mock()
            result = await service._run_subagent(skill, "test", progress_callback=callback, subagent_id="sa-1")

        assert result.success is False
        calls = [c.args[0] for c in callback.call_args_list]
        assert any("[sa-1]" in c and "started" in c for c in calls)
        assert any("[sa-1]" in c and "failed" in c for c in calls)

    @pytest.mark.asyncio
    async def test_callback_on_timeout(self):
        """Progress callback fires when subagent times out."""
        mock_client = MagicMock()

        async def slow_create(**kwargs):
            import asyncio
            await asyncio.sleep(10)

        mock_client.messages.create = slow_create
        mock_executor = MagicMock()
        callback = MagicMock()

        service = SubagentService(mock_client, mock_executor)

        tasks = [{"skill": "literature_review", "query": "test"}]

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            settings = self._make_settings_mock()
            settings.subagent_timeout = 0.1  # very short timeout
            mock_settings.return_value = settings
            await service.run_subagents(tasks, progress_callback=callback)

        calls = [c.args[0] for c in callback.call_args_list]
        assert any("[sa-1]" in c and "timed out" in c for c in calls)


class TestFormatToolParams:
    """Tests for the _format_tool_params helper."""

    def test_empty_dict(self):
        assert _format_tool_params({}) == ""

    def test_string_values(self):
        result = _format_tool_params({"gene": "BRCA1", "species": "human"})
        assert result == "(gene='BRCA1', species='human')"

    def test_non_string_values(self):
        result = _format_tool_params({"limit": 10, "verbose": True})
        assert result == "(limit=10, verbose=True)"

    def test_long_string_truncated(self):
        long_val = "A" * 100
        result = _format_tool_params({"query": long_val})
        assert "..." in result
        assert len(result.split("'")[1]) < 100

    def test_list_value(self):
        result = _format_tool_params({"ids": [1, 2, 3]})
        assert result == "(ids=<list>)"

    def test_dict_value(self):
        result = _format_tool_params({"filter": {"key": "val"}})
        assert result == "(filter=<dict>)"

    def test_max_len_truncation(self):
        result = _format_tool_params(
            {"a": "short", "b": "short", "c": "short", "d": "short"},
            max_len=20,
        )
        assert len(result) <= 20
        assert result.endswith("...")

    def test_mixed_types(self):
        result = _format_tool_params({"gene": "TP53", "limit": 5, "data": [1]})
        assert "gene='TP53'" in result
        assert "limit=5" in result
        assert "data=<list>" in result


class TestExternalToolInclusion:
    """Tests that _get_tool_definitions includes external tools when include_external is True."""

    def test_external_tools_included_when_flag_set(self):
        mock_client = MagicMock()
        mock_executor = MagicMock()
        service = SubagentService(mock_client, mock_executor)

        skill = get_skill("genetics_data_extraction")
        assert skill is not None

        fake_external = [{"name": "ext_tool_1", "description": "ext", "input_schema": {}}]

        with (
            patch("genetics_mcp_server.subagent.get_settings") as mock_settings,
            patch("genetics_mcp_server.subagent.get_external_anthropic_tools", return_value=fake_external),
            patch.object(skill, "include_external", True),
        ):
            settings = MagicMock()
            settings.disabled_tools = set()
            settings.enable_subagents = True
            settings.enable_script_execution = False
            settings.subagent_allowed_paths_list = []
            mock_settings.return_value = settings

            tools = service._get_tool_definitions(skill)

        tool_names = [t["name"] for t in tools]
        assert "ext_tool_1" in tool_names

    def test_external_tools_excluded_when_flag_not_set(self):
        mock_client = MagicMock()
        mock_executor = MagicMock()
        service = SubagentService(mock_client, mock_executor)

        skill = get_skill("literature_review")
        assert skill is not None

        with (
            patch("genetics_mcp_server.subagent.get_settings") as mock_settings,
            patch("genetics_mcp_server.subagent.get_external_anthropic_tools") as mock_ext,
        ):
            settings = MagicMock()
            settings.disabled_tools = set()
            settings.enable_subagents = True
            settings.enable_script_execution = False
            settings.subagent_allowed_paths_list = []
            mock_settings.return_value = settings

            # ensure include_external is False
            assert not skill.include_external

            service._get_tool_definitions(skill)

        mock_ext.assert_not_called()


class TestSubagentIdInResults:
    """Tests that subagent_id propagates through to run_subagents output."""

    def _make_settings_mock(self):
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
        return settings

    def _make_message(self, text="done"):
        content = []
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        content.append(text_block)

        msg = MagicMock()
        msg.content = content
        msg.stop_reason = "end_turn"
        msg.usage = MagicMock()
        msg.usage.input_tokens = 10
        msg.usage.output_tokens = 5
        return msg

    @pytest.mark.asyncio
    async def test_subagent_id_in_run_subagent_result(self):
        msg = self._make_message("result")
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=msg)
        mock_executor = MagicMock()

        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("literature_review")

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            mock_settings.return_value = self._make_settings_mock()
            result = await service._run_subagent(skill, "test", subagent_id="sa-5")

        assert result.subagent_id == "sa-5"

    @pytest.mark.asyncio
    async def test_subagent_ids_in_parallel_results(self):
        """run_subagents assigns sequential sa-N IDs to each task."""
        msg = self._make_message("done")
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=msg)
        mock_executor = MagicMock()

        service = SubagentService(mock_client, mock_executor)
        tasks = [
            {"skill": "literature_review", "query": "q1"},
            {"skill": "genetics_data_extraction", "query": "q2"},
        ]

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            mock_settings.return_value = self._make_settings_mock()
            result = await service.run_subagents(tasks)

        assert result["success"] is True
        assert result["results"][0]["subagent_id"] == "sa-1"
        assert result["results"][1]["subagent_id"] == "sa-2"
