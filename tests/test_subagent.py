"""Tests for the subagent system: skills, sandbox tools, and subagent service."""

import json
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
    get_sandbox_tool_definitions,
    list_directory,
    read_file,
)
from genetics_mcp_server.subagent import SubagentResult, SubagentService, _format_tool_params
from genetics_mcp_server.tools.definitions import get_anthropic_tools


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
        assert "database_analysis" in desc
        assert "data_analysis" in desc

    def test_declared_tools_exist(self):
        """A skill may only name tools that exist, or the name is silently a no-op."""
        known = {t["name"] for t in get_anthropic_tools()}
        for skill in SKILL_REGISTRY.values():
            unknown = skill.tools - known
            assert not unknown, f"Skill '{skill.name}' names unknown tools: {unknown}"

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


class TestSandboxToolDefinitions:
    """Tests for sandbox tool definition generation."""

    def test_no_tools_when_read_disabled(self):
        assert get_sandbox_tool_definitions(False) == []

    def test_file_tools_when_read_enabled(self):
        names = [t["name"] for t in get_sandbox_tool_definitions(True)]
        assert names == ["read_file", "list_directory"]


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
            settings.enable_subagents = True
            settings.disabled_tools = set()
            settings.subagent_allowed_paths_list = []
            mock_settings.return_value = settings

            result = await service._run_subagent(skill, "Find papers about PCSK9", user=None, session_id=None, gateway_asserted=False)

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
            settings.enable_subagents = True
            settings.disabled_tools = set()
            settings.subagent_allowed_paths_list = []
            mock_settings.return_value = settings

            result = await service.run_subagents(tasks, user=None, session_id=None, gateway_asserted=False)

        assert result["success"] is True
        assert len(result["results"]) == 2
        assert all(r["success"] for r in result["results"])

    @pytest.mark.asyncio
    async def test_unknown_skill_returns_error(self):
        mock_client = MagicMock()
        mock_executor = MagicMock()
        service = SubagentService(mock_client, mock_executor)

        result = await service.run_subagents([{"skill": "nonexistent", "query": "test"}], user=None, session_id=None, gateway_asserted=False)
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
                settings.subagent_allowed_paths_list = []
                mock_settings.return_value = settings

                tools = service._get_tool_definitions(skill)

            tool_names = [t["name"] for t in tools]
            assert "launch_subagents" not in tool_names, (
                f"launch_subagents should be excluded for skill '{skill_name}'"
            )

    def test_artifact_tools_excluded_from_every_skill(self):
        """read_artifact and list_capabilities stay off every skill.

        read_artifact resolves a model-supplied NAME against the executions one
        (user, session) pair ran, and every subagent of one turn shares that pair, so a
        subagent could otherwise reach an artifact another execution wrote. run_analysis is
        a different case — it is minted against the identity the caller threads in — and is
        checked on its own below.
        """
        mock_client = MagicMock()
        mock_executor = MagicMock()
        service = SubagentService(mock_client, mock_executor)

        for skill_name in SKILL_REGISTRY:
            skill = get_skill(skill_name)
            with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
                settings = MagicMock()
                settings.disabled_tools = set()
                settings.enable_subagents = True
                settings.subagent_allowed_paths_list = []
                mock_settings.return_value = settings

                tools = service._get_tool_definitions(skill)

            tool_names = {t["name"] for t in tools}
            assert "read_artifact" not in tool_names, (
                f"read_artifact should be excluded for skill '{skill_name}'"
            )
            assert "list_capabilities" not in tool_names, (
                f"list_capabilities should be excluded for skill '{skill_name}'"
            )


class TestSubagentDispatchAllowList:
    """The dispatcher refuses any tool the skill's own definitions did not declare.

    Excluding a tool from _get_tool_definitions only stops it being *offered*. The local
    branch of _execute_subagent_tool resolves the model's tool_name against the executor
    by getattr, so without an allow-list a subagent could call an excluded tool anyway by
    naming it — and the subagent's task text is written by the parent model, which does
    have run_analysis declared.
    """

    def _settings(self, mock_settings):
        settings = MagicMock()
        settings.disabled_tools = set()
        settings.enable_subagents = True
        settings.subagent_allowed_paths_list = []
        mock_settings.return_value = settings

    @pytest.mark.asyncio
    async def test_run_analysis_forgery_never_reaches_handler(self):
        """A model-supplied identity on an excluded tool must not reach the executor.

        run_analysis passes `user`/`session_id` to mint_execution_tokens, where they
        become the sub/sid of both per-execution JWTs and of every audit record.
        """
        mock_client = MagicMock()
        mock_executor = MagicMock()
        mock_executor.run_analysis = AsyncMock(return_value={"success": True})
        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("genetics_data_extraction")

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            self._settings(mock_settings)
            result = await service._execute_subagent_tool(
                "run_analysis",
                {
                    "code": "print(1)",
                    "user": "attacker@evil.example",
                    "session_id": "other-sid",
                },
                skill, user=None, session_id=None, gateway_asserted=False
            )

        mock_executor.run_analysis.assert_not_awaited()
        assert result["success"] is False
        assert "run_analysis" in result["error"]

    @pytest.mark.asyncio
    async def test_undeclared_executor_attribute_refused(self):
        """getattr on the executor is not an authorization decision."""
        mock_client = MagicMock()
        mock_executor = MagicMock()
        mock_executor.close = AsyncMock(return_value={"success": True})
        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("genetics_data_extraction")

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            self._settings(mock_settings)
            result = await service._execute_subagent_tool("close", {}, skill, user=None, session_id=None, gateway_asserted=False)

        mock_executor.close.assert_not_awaited()
        assert result["success"] is False
        assert "not available to this subagent" in result["error"]

    @pytest.mark.asyncio
    async def test_subagent_cannot_name_the_user_it_executes_as(self):
        """A skill that CAN run code still cannot choose the subject it runs under.

        data_analysis declares run_analysis, so the allow-list lets the call through; what
        must not get through is the `user`/`session_id` the model wrote, because those
        become the sub/sid of both per-execution JWTs and of every audit record.
        """
        mock_client = MagicMock()
        mock_executor = MagicMock()
        mock_executor.run_analysis = AsyncMock(return_value={"success": True})
        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("data_analysis")

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            self._settings(mock_settings)
            await service._execute_subagent_tool(
                "run_analysis",
                {
                    "code": "print(1)",
                    "user": "attacker@evil.example",
                    "session_id": "someone-elses-session",
                    "gateway_asserted": True,
                },
                skill,
                user="real@example.org",
                session_id="sess-real",
                gateway_asserted=False,
            )

        kwargs = mock_executor.run_analysis.await_args.kwargs
        assert kwargs["user"] == "real@example.org"
        assert kwargs["session_id"] == "sess-real"
        assert kwargs["gateway_asserted"] is False

    @pytest.mark.asyncio
    async def test_missing_identity_is_not_replaced_by_model_input(self):
        """A caller that threads nothing hands run_analysis nothing, not the model's guess.

        run_analysis refuses without an authenticated pair; that refusal is only reachable
        if the forged pair was dropped rather than passed on.
        """
        mock_client = MagicMock()
        mock_executor = MagicMock()
        mock_executor.run_analysis = AsyncMock(return_value={"success": True})
        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("data_analysis")

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            self._settings(mock_settings)
            await service._execute_subagent_tool(
                "run_analysis",
                {"code": "print(1)", "user": "attacker@evil.example", "session_id": "x"},
                skill, user=None, session_id=None, gateway_asserted=False
            )

        kwargs = mock_executor.run_analysis.await_args.kwargs
        assert kwargs["user"] is None
        assert kwargs["session_id"] is None

    @pytest.mark.asyncio
    async def test_declared_tool_still_dispatches(self):
        """The guard must not break a skill calling a tool its definitions do produce."""
        mock_client = MagicMock()
        mock_executor = MagicMock()
        mock_executor.list_datasets = AsyncMock(return_value={"success": True, "datasets": []})
        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("genetics_data_extraction")

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            self._settings(mock_settings)
            result = await service._execute_subagent_tool("list_datasets", {}, skill, user=None, session_id=None, gateway_asserted=False)

        mock_executor.list_datasets.assert_awaited_once()
        assert result["success"] is True


class TestIdentityThreading:
    """The identity reaches the dispatch from run_subagents, never from the task text."""

    @staticmethod
    def _settings(mock_settings):
        settings = MagicMock()
        settings.disabled_tools = set()
        settings.enable_subagents = True
        settings.subagent_allowed_paths_list = []
        settings.subagent_timeout = 30
        settings.subagent_model = ""
        settings.fast_model = "claude-haiku-4-5"
        settings.temperature = None
        settings.max_continuations = 1
        settings.mcp_max_result_size = 50000
        mock_settings.return_value = settings

    @pytest.mark.asyncio
    async def test_run_subagents_threads_identity_to_the_dispatch(self):
        """The whole point of the task: the pair the request authenticated reaches the sandbox."""
        svc = TestSubagentService()
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=[
                svc._make_mock_message(
                    "running",
                    tool_uses=[{
                        "id": "tu-1",
                        "name": "run_analysis",
                        # the model names an identity of its own; it must be discarded
                        "input": {"code": "print(1)", "user": "spoofed@evil.example"},
                    }],
                ),
                svc._make_mock_message("done"),
            ]
        )
        mock_executor = MagicMock()
        mock_executor.run_analysis = AsyncMock(return_value={"success": True})
        service = SubagentService(mock_client, mock_executor)

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            self._settings(mock_settings)
            result = await service.run_subagents(
                [{"skill": "data_analysis", "query": "analyse this"}],
                user="real@example.org",
                session_id="sess-real",
                gateway_asserted=True,
            )

        assert result["success"] is True
        kwargs = mock_executor.run_analysis.await_args.kwargs
        assert kwargs["user"] == "real@example.org"
        assert kwargs["session_id"] == "sess-real"
        assert kwargs["gateway_asserted"] is True

    @pytest.mark.asyncio
    async def test_image_artifacts_are_not_returned_to_the_subagent(self):
        """base64 a subagent can neither see nor display must not reach its context."""
        mock_client = MagicMock()
        mock_executor = MagicMock()
        mock_executor.run_analysis = AsyncMock(
            return_value={
                "success": True,
                "images": [{"name": "plot.png", "content_base64": "A" * 5000}],
            }
        )
        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("data_analysis")

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            self._settings(mock_settings)
            result = await service._execute_subagent_tool(
                "run_analysis", {"code": "print(1)"}, skill, user="u@x", session_id="s", gateway_asserted=False
            )

        assert "images" not in result
        assert "plot.png" in result["note"]

    @pytest.mark.asyncio
    async def test_artifacts_note_written_for_the_main_path_is_dropped(self):
        """Neither half of run_analysis's artifacts_note is true for a subagent."""
        mock_client = MagicMock()
        mock_executor = MagicMock()
        mock_executor.run_analysis = AsyncMock(
            return_value={
                "success": True,
                "artifacts": [{"name": "plot.png"}, {"name": "table.csv"}],
                "artifacts_note": (
                    "Image artifacts have been displayed to the user already; describe "
                    "what the plot shows. Read any other artifact with read_artifact."
                ),
            }
        )
        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("data_analysis")

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            settings = MagicMock()
            settings.disabled_tools = set()
            settings.enable_subagents = True
            settings.subagent_allowed_paths_list = []
            mock_settings.return_value = settings
            result = await service._execute_subagent_tool(
                "run_analysis",
                {"code": "print(1)"},
                skill,
                user="u@x",
                session_id="s",
                gateway_asserted=False,
            )

        assert "artifacts_note" not in result
        rendered = json.dumps(result)
        assert "displayed to the user" not in rendered
        assert "read_artifact" not in rendered


class TestDownloadHintProcessing:
    """Tests that _execute_subagent_tool applies _process_download_hints on local tool results."""

    @pytest.mark.asyncio
    async def test_download_hints_called_on_local_tool_result(self):
        """Local tool results should be processed through _process_download_hints."""
        mock_client = MagicMock()
        mock_executor = MagicMock()
        raw_result = {"success": True, "data": "test"}
        mock_executor.list_datasets = AsyncMock(return_value=raw_result)

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

            result = await service._execute_subagent_tool("list_datasets", {}, skill, user=None, session_id=None, gateway_asserted=False)

        # the tool name is passed so a download failure names its producer in the log
        mock_hints.assert_called_once_with(raw_result, owner=None, tool_name="list_datasets")
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
        mock_executor.search_genes = AsyncMock(return_value=raw_result)

        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("genetics_data_extraction")

        with (
            patch("genetics_mcp_server.subagent.get_settings") as mock_settings,
            patch("genetics_mcp_server.subagent.is_external_tool", return_value=False),
        ):
            settings = MagicMock()
            settings.subagent_allowed_paths_list = []
            mock_settings.return_value = settings

            result = await service._execute_subagent_tool("search_genes", {}, skill, user=None, session_id=None, gateway_asserted=False)

        assert "INCLUDE_IN_RESPONSE" in result
        assert "Download" in result["INCLUDE_IN_RESPONSE"]
        assert "_download_url" not in result

    @pytest.mark.asyncio
    async def test_no_download_hints_on_failure(self):
        """Failed results should pass through without download hint processing."""
        mock_client = MagicMock()
        mock_executor = MagicMock()
        raw_result = {"success": False, "error": "something broke"}
        mock_executor.search_phenotypes = AsyncMock(return_value=raw_result)

        service = SubagentService(mock_client, mock_executor)
        skill = get_skill("genetics_data_extraction")

        with (
            patch("genetics_mcp_server.subagent.get_settings") as mock_settings,
            patch("genetics_mcp_server.subagent.is_external_tool", return_value=False),
        ):
            settings = MagicMock()
            settings.subagent_allowed_paths_list = []
            mock_settings.return_value = settings

            result = await service._execute_subagent_tool("search_phenotypes", {}, skill, user=None, session_id=None, gateway_asserted=False)

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
        settings.enable_subagents = True
        settings.disabled_tools = set()
        settings.subagent_allowed_paths_list = []
        settings.max_continuations = 3
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
            result = await service._run_subagent(skill, "test query", user=None, session_id=None, gateway_asserted=False)

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
            result = await service._run_subagent(skill, "test query", user=None, session_id=None, gateway_asserted=False)

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
            result = await service.run_subagents(tasks, user=None, session_id=None, gateway_asserted=False)

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
        settings.enable_subagents = True
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
            await service._run_subagent(skill, "test", progress_callback=callback, subagent_id="sa-1", user=None, session_id=None, gateway_asserted=False)

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
            await service._run_subagent(skill, "test", progress_callback=callback, subagent_id="sa-2", user=None, session_id=None, gateway_asserted=False)

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
            result = await service._run_subagent(skill, "test", progress_callback=callback, subagent_id="sa-1", user=None, session_id=None, gateway_asserted=False)

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
            await service.run_subagents(tasks, progress_callback=callback, user=None, session_id=None, gateway_asserted=False)

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
        settings.enable_subagents = True
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
            result = await service._run_subagent(skill, "test", subagent_id="sa-5", user=None, session_id=None, gateway_asserted=False)

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
            result = await service.run_subagents(tasks, user=None, session_id=None, gateway_asserted=False)

        assert result["success"] is True
        assert result["results"][0]["subagent_id"] == "sa-1"
        assert result["results"][1]["subagent_id"] == "sa-2"


class TestSubagentTruncation:
    """A report stopped by max_tokens must not be returned as complete findings."""

    def _settings(self, max_continuations=3, subagent_model=""):
        settings = MagicMock()
        settings.subagent_model = subagent_model
        settings.fast_model = "claude-haiku-4-5"
        settings.temperature = None
        settings.mcp_max_result_size = 50000
        settings.subagent_timeout = 120
        settings.enable_subagents = True
        settings.disabled_tools = set()
        settings.subagent_allowed_paths_list = []
        settings.max_continuations = max_continuations
        return settings

    def _message(self, text, stop_reason):
        block = MagicMock()
        block.type = "text"
        block.text = text
        block.model_dump.return_value = {"type": "text", "text": text}
        msg = MagicMock()
        msg.content = [block]
        msg.stop_reason = stop_reason
        msg.usage.input_tokens = 100
        msg.usage.output_tokens = 50
        return msg

    @pytest.mark.asyncio
    async def test_truncated_report_is_continued(self):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=[
                self._message("first half", "max_tokens"),
                self._message(" second half", "end_turn"),
            ]
        )
        service = SubagentService(mock_client, MagicMock())

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            mock_settings.return_value = self._settings()
            result = await service._run_subagent(get_skill("literature_review"), "q", user=None, session_id=None, gateway_asserted=False)

        assert result.success is True
        assert result.truncated is False
        assert result.output == "first half second half"

        # the resume request must end on a user turn: a trailing assistant message
        # is a prefill, which Opus 4.6+ rejects
        resume_messages = mock_client.messages.create.await_args_list[1].kwargs["messages"]
        assert resume_messages[-1]["role"] == "user"
        assert resume_messages[-2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_truncation_is_bounded_and_marked(self):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=[self._message(f"part{i} ", "max_tokens") for i in range(3)]
        )
        service = SubagentService(mock_client, MagicMock())

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            mock_settings.return_value = self._settings(max_continuations=2)
            result = await service._run_subagent(get_skill("literature_review"), "q", user=None, session_id=None, gateway_asserted=False)

        assert result.truncated is True
        assert "[TRUNCATED:" in result.output
        # initial turn + 2 continuations, then it gives up
        assert mock_client.messages.create.await_count == 3

    @pytest.mark.asyncio
    async def test_thinking_set_only_for_supporting_models(self):
        for model, expected in [("claude-opus-5", True), ("claude-haiku-4-5", False)]:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(
                return_value=self._message("done", "end_turn")
            )
            service = SubagentService(mock_client, MagicMock())

            with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
                mock_settings.return_value = self._settings(subagent_model=model)
                await service._run_subagent(get_skill("literature_review"), "q", user=None, session_id=None, gateway_asserted=False)

            params = mock_client.messages.create.await_args.kwargs
            assert ("thinking" in params) is expected, model

    @pytest.mark.asyncio
    async def test_truncated_flag_reaches_the_main_agent(self):
        """The main agent sees the flag in the tool result, not just in the log."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=[self._message("cut ", "max_tokens")] * 4
        )
        service = SubagentService(mock_client, MagicMock())

        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            mock_settings.return_value = self._settings(max_continuations=3)
            payload = await service.run_subagents(
                [{"skill": "literature_review", "query": "q"}], user=None, session_id=None, gateway_asserted=False
            )

        assert payload["results"][0]["truncated"] is True


# The exact tool names each skill resolves to. Pinned rather than derived: a skill's surface
# used to be a function of tool `category` and the tool_profile table, so retuning either for
# the main agent silently changed what a subagent could reach. A failure here means a skill
# gained or lost a tool — decide whether that was intended, then update this map.
_PINNED_SKILL_TOOLS: dict[str, set[str]] = {
    "genetics_data_extraction": {
        "analyze_variant_list",
        "get_asm_qtl_by_gene",
        "get_asm_qtl_by_variant",
        "get_colocalization",
        "get_colocalization_by_credible_set",
        "get_credible_set_by_id",
        "get_credible_set_leads_by_phenotype",
        "get_credible_sets_by_gene",
        "get_credible_sets_by_phenotype",
        "get_credible_sets_by_qtl_gene",
        "get_credible_sets_by_region",
        "get_credible_sets_by_variant",
        "get_credible_sets_stats",
        "get_dataset_display_names",
        "get_drug_profile",
        "get_drug_targets_for_gene",
        "get_exome_results_by_gene",
        "get_exome_results_by_phenotype",
        "get_exome_results_by_region",
        "get_exome_results_by_variant",
        "get_gene_based_results",
        "get_gene_based_results_by_phenotype",
        "get_gene_disease_associations",
        "get_gene_expression",
        "get_gene_group_members",
        "get_gene_to_peaks",
        "get_genes_in_region",
        "get_hla_by_allele",
        "get_hla_by_phenotype",
        "get_ld_between_variants",
        "get_mpra_by_gene",
        "get_mpra_by_region",
        "get_mpra_by_variant",
        "get_mpra_pip_concordance_by_gene",
        "get_myvariant_annotations",
        "get_nearest_genes",
        "get_open_chromatin_by_gene",
        "get_open_chromatin_by_peak",
        "get_open_chromatin_by_region",
        "get_open_chromatin_by_variant",
        "get_peak_to_genes",
        "get_phenotype_report",
        "get_protein_annotations",
        "get_resource_metadata",
        "get_summary_stats",
        "get_summary_stats_by_region",
        "get_target_bioactivity",
        "get_variant_annotations",
        "get_variant_effect_by_gene",
        "get_variant_effect_by_variant",
        "get_variant_protein_effect",
        "get_variants_in_ld",
        "list_datasets",
        "lookup_phenotype_names",
        "lookup_variants_by_rsid",
        "map_protein_variants",
        "normalize_gene_symbols",
        "search_cbioportal",
        "search_genes",
        "search_mgi",
        "search_phenotypes",
        "search_scientific_literature",
        "search_uniprot",
        "web_search",
    },
    "literature_review": {
        "get_dataset_display_names",
        "get_drug_profile",
        "get_drug_targets_for_gene",
        "get_gene_group_members",
        "get_protein_annotations",
        "get_resource_metadata",
        "get_target_bioactivity",
        "get_variant_protein_effect",
        "list_datasets",
        "lookup_phenotype_names",
        "lookup_variants_by_rsid",
        "map_protein_variants",
        "normalize_gene_symbols",
        "search_cbioportal",
        "search_genes",
        "search_mgi",
        "search_phenotypes",
        "search_scientific_literature",
        "search_uniprot",
        "web_search",
    },
    "database_analysis": {
        "get_database_schema",
        "get_dataset_display_names",
        "get_drug_profile",
        "get_drug_targets_for_gene",
        "get_gene_group_members",
        "get_protein_annotations",
        "get_resource_metadata",
        "get_target_bioactivity",
        "get_variant_protein_effect",
        "list_datasets",
        "lookup_phenotype_names",
        "lookup_variants_by_rsid",
        "map_protein_variants",
        "normalize_gene_symbols",
        "query_database",
        "search_cbioportal",
        "search_genes",
        "search_mgi",
        "search_phenotypes",
        "search_scientific_literature",
        "search_uniprot",
        "web_search",
    },
    "variant_list_analysis": {
        "analyze_variant_list",
        "get_asm_qtl_by_gene",
        "get_asm_qtl_by_variant",
        "get_colocalization",
        "get_colocalization_by_credible_set",
        "get_credible_set_by_id",
        "get_credible_set_leads_by_phenotype",
        "get_credible_sets_by_gene",
        "get_credible_sets_by_phenotype",
        "get_credible_sets_by_qtl_gene",
        "get_credible_sets_by_region",
        "get_credible_sets_by_variant",
        "get_credible_sets_stats",
        "get_dataset_display_names",
        "get_drug_profile",
        "get_drug_targets_for_gene",
        "get_exome_results_by_gene",
        "get_exome_results_by_phenotype",
        "get_exome_results_by_region",
        "get_exome_results_by_variant",
        "get_gene_based_results",
        "get_gene_based_results_by_phenotype",
        "get_gene_disease_associations",
        "get_gene_expression",
        "get_gene_group_members",
        "get_gene_to_peaks",
        "get_genes_in_region",
        "get_hla_by_allele",
        "get_hla_by_phenotype",
        "get_ld_between_variants",
        "get_mpra_by_gene",
        "get_mpra_by_region",
        "get_mpra_by_variant",
        "get_mpra_pip_concordance_by_gene",
        "get_myvariant_annotations",
        "get_nearest_genes",
        "get_open_chromatin_by_gene",
        "get_open_chromatin_by_peak",
        "get_open_chromatin_by_region",
        "get_open_chromatin_by_variant",
        "get_peak_to_genes",
        "get_phenotype_report",
        "get_protein_annotations",
        "get_resource_metadata",
        "get_summary_stats",
        "get_summary_stats_by_region",
        "get_target_bioactivity",
        "get_variant_annotations",
        "get_variant_effect_by_gene",
        "get_variant_effect_by_variant",
        "get_variant_protein_effect",
        "get_variants_in_ld",
        "list_datasets",
        "lookup_phenotype_names",
        "lookup_variants_by_rsid",
        "map_protein_variants",
        "normalize_gene_symbols",
        "search_cbioportal",
        "search_genes",
        "search_mgi",
        "search_phenotypes",
        "search_scientific_literature",
        "search_uniprot",
        "web_search",
    },
    "data_analysis": {
        "get_dataset_display_names",
        "get_drug_profile",
        "get_drug_targets_for_gene",
        "get_gene_group_members",
        "get_protein_annotations",
        "get_resource_metadata",
        "get_target_bioactivity",
        "get_variant_protein_effect",
        "list_datasets",
        "list_directory",
        "lookup_phenotype_names",
        "lookup_variants_by_rsid",
        "map_protein_variants",
        "normalize_gene_symbols",
        "read_file",
        "run_analysis",
        "search_cbioportal",
        "search_genes",
        "search_mgi",
        "search_phenotypes",
        "search_scientific_literature",
        "search_uniprot",
        "web_search",
    },
}


class TestSkillToolSurface:
    """Pins the resolved tool set of every skill."""

    def _resolve(self, skill):
        service = SubagentService(MagicMock(), MagicMock())
        with patch("genetics_mcp_server.subagent.get_settings") as mock_settings:
            settings = MagicMock()
            settings.disabled_tools = set()
            settings.enable_subagents = True
            mock_settings.return_value = settings
            return {t["name"] for t in service._get_tool_definitions(skill)}

    def test_every_skill_is_pinned(self):
        assert set(_PINNED_SKILL_TOOLS) == set(SKILL_REGISTRY)

    @pytest.mark.parametrize("skill_name", sorted(_PINNED_SKILL_TOOLS))
    def test_resolved_tools_match_pin(self, skill_name):
        resolved = self._resolve(SKILL_REGISTRY[skill_name])
        expected = _PINNED_SKILL_TOOLS[skill_name]
        assert resolved == expected, (
            f"{skill_name} lost {sorted(expected - resolved)} "
            f"and gained {sorted(resolved - expected)}"
        )

    def test_no_skill_can_reach_orchestration(self):
        """Recursive launches and another execution's artifacts stay off every skill.

        run_analysis has left this set — it is threaded an authenticated identity and one
        skill declares it — so the two that remain are asserted for all five skills, and
        run_analysis's single-skill reach is pinned below.
        """
        forbidden = {"launch_subagents", "read_artifact", "list_capabilities"}
        for skill_name, skill in SKILL_REGISTRY.items():
            assert not self._resolve(skill) & forbidden, skill_name

    def test_only_data_analysis_can_execute(self):
        """Exactly one skill runs code, and it is the one whose instructions say so."""
        can_execute = {
            name
            for name, skill in SKILL_REGISTRY.items()
            if "run_analysis" in self._resolve(skill)
        }
        assert can_execute == {"data_analysis"}

    def test_data_analysis_runs_its_own_script(self):
        """The data_analysis skill drafts a script AND runs it."""
        resolved = self._resolve(SKILL_REGISTRY["data_analysis"])
        assert resolved & {"read_file", "list_directory"} == {"read_file", "list_directory"}
        assert "run_analysis" in resolved
        assert not any("exec" in name for name in resolved)
