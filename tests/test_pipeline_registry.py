"""Tests for pipeline templates and registry."""

import pytest
from feature_prd_runner.pipelines.registry import (
    BUILTIN_TEMPLATES,
    PipelineRegistry,
    PipelineTemplate,
    StepDef,
)


class TestPipelineTemplate:
    def test_builtin_templates_exist(self):
        expected = {"feature", "bug_fix", "refactor", "research", "docs", "test", "repo_review", "security_audit"}
        assert expected == set(BUILTIN_TEMPLATES.keys())

    def test_feature_pipeline_steps(self):
        tmpl = BUILTIN_TEMPLATES["feature"]
        assert tmpl.step_names() == ["plan", "plan_impl", "implement", "verify", "review", "commit"]
        assert tmpl.task_types == ("feature",)

    def test_bug_fix_pipeline_steps(self):
        tmpl = BUILTIN_TEMPLATES["bug_fix"]
        assert "reproduce" in tmpl.step_names()
        assert "diagnose" in tmpl.step_names()

    def test_research_pipeline_steps(self):
        tmpl = BUILTIN_TEMPLATES["research"]
        assert tmpl.step_names() == ["gather", "analyze", "summarize", "report"]

    def test_step_names(self):
        tmpl = PipelineTemplate(
            id="test",
            display_name="Test",
            description="Test pipeline",
            steps=(
                StepDef(name="a"),
                StepDef(name="b"),
                StepDef(name="c"),
            ),
        )
        assert tmpl.step_names() == ["a", "b", "c"]


class TestPipelineRegistry:
    def test_list_templates(self):
        reg = PipelineRegistry()
        templates = reg.list_templates()
        assert len(templates) == 8

    def test_get_template(self):
        reg = PipelineRegistry()
        tmpl = reg.get("feature")
        assert tmpl.id == "feature"

    def test_get_unknown_raises(self):
        reg = PipelineRegistry()
        with pytest.raises(KeyError, match="Unknown pipeline"):
            reg.get("nonexistent")

    def test_resolve_for_task_type(self):
        reg = PipelineRegistry()
        assert reg.resolve_for_task_type("feature").id == "feature"
        assert reg.resolve_for_task_type("bug").id == "bug_fix"
        assert reg.resolve_for_task_type("refactor").id == "refactor"
        assert reg.resolve_for_task_type("research").id == "research"
        assert reg.resolve_for_task_type("docs").id == "docs"
        assert reg.resolve_for_task_type("test").id == "test"

    def test_resolve_unknown_type_defaults_to_feature(self):
        reg = PipelineRegistry()
        assert reg.resolve_for_task_type("unknown_type").id == "feature"

    def test_register_custom(self):
        reg = PipelineRegistry()
        custom = PipelineTemplate(
            id="custom",
            display_name="Custom",
            description="Custom pipeline",
            steps=(StepDef(name="plan"), StepDef(name="implement")),
            task_types=("custom_type",),
        )
        reg.register(custom)
        assert reg.get("custom").id == "custom"
        assert reg.resolve_for_task_type("custom_type").id == "custom"

    def test_unregister(self):
        reg = PipelineRegistry()
        reg.unregister("research")
        with pytest.raises(KeyError):
            reg.get("research")
