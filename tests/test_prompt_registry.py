from hashlib import md5


def test_prompt_registry_render_returns_text_and_stable_md5_version():
    from src.prompts.registry import prompt_registry

    rendered, version = prompt_registry.render(
        "summarize",
        {
            "query": "LangGraph",
            "source_blocks": "SOURCE URL: https://a.com\nSOURCE TITLE: A\nCONTENT:\nAlpha",
            "domain": "",
        },
    )

    assert "LangGraph" in rendered
    assert version == md5(
        (prompt_registry.template_dir / "summarize.j2").read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()


def test_prompt_registry_reuses_cached_template_object():
    from src.prompts.registry import prompt_registry

    prompt_registry._cache.clear()
    prompt_registry.render("summarize", {"query": "Q", "source_blocks": "S", "domain": ""})
    first_template = prompt_registry._cache["summarize"]
    prompt_registry.render("summarize", {"query": "Q2", "source_blocks": "S2", "domain": ""})
    second_template = prompt_registry._cache["summarize"]

    assert first_template is second_template


def test_prompt_registry_versions_differ_for_different_templates():
    from src.prompts.registry import prompt_registry

    _, summarize_version = prompt_registry.render(
        "summarize", {"query": "Q", "source_blocks": "S", "domain": ""}
    )
    _, report_version = prompt_registry.render(
        "report",
        {
            "query": "Q",
            "summaries_text": "Summary",
            "memory_context": "",
            "domain": "",
        },
    )

    assert summarize_version != report_version
