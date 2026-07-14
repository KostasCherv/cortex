"""Tests for citation provenance and final citation selection."""

from src.citations import (
    build_rag_citations,
    build_web_citations,
    select_chat_citations,
)


def test_build_rag_citations_includes_source_type():
    citations = build_rag_citations(
        [
            {
                "source_title": "brief.pdf",
                "source_url": "https://example.com/brief.pdf",
                "chunk_id": "chunk-1",
                "text": "Important detail",
            }
        ]
    )
    assert citations == [
        {
            "source_title": "brief.pdf",
            "source_url": "https://example.com/brief.pdf",
            "chunk_id": "chunk-1",
            "text": "Important detail",
            "source_type": "rag",
        }
    ]


def test_build_web_citations_uses_content_fallback():
    citations = build_web_citations(
        [
            {
                "title": "Crypto Daily",
                "url": "https://example.com/crypto",
                "content": "Bitcoin rises on ETF flows",
            }
        ],
        "tavily",
    )
    assert citations == [
        {
            "source_title": "Crypto Daily",
            "source_url": "https://example.com/crypto",
            "chunk_id": "tavily-web-1",
            "text": "Bitcoin rises on ETF flows",
            "source_type": "web",
        }
    ]


def test_select_chat_citations_prefers_web_when_web_used():
    rag_chunks = [
        {
            "source_title": "SaaS_Starter_Kit.pdf",
            "source_url": "",
            "chunk_id": "rag-1",
            "text": "starter kit",
        },
        {
            "source_title": "The-Founders-Playbook.pdf",
            "source_url": "",
            "chunk_id": "rag-2",
            "text": "playbook",
        },
    ]
    web_citations = [
        {
            "source_title": "Crypto Daily",
            "source_url": "https://example.com/crypto",
            "chunk_id": "tavily-web-1",
            "text": "Bitcoin rises",
            "source_type": "web",
        }
    ]

    citations = select_chat_citations(
        rag_chunks,
        web_citations,
        router_action="web_search",
        web_used=True,
        rag_context_text="irrelevant workspace context",
    )

    assert citations == web_citations
    assert "SaaS_Starter_Kit.pdf" not in {row["source_title"] for row in citations}


def test_select_chat_citations_uses_rag_when_no_tool_citations():
    rag_chunks = [
        {
            "source_title": "brief.pdf",
            "source_url": "",
            "chunk_id": "rag-1",
            "text": "quarterly results",
        }
    ]

    citations = select_chat_citations(
        rag_chunks,
        [],
        router_action="answer_from_rag",
        web_used=False,
        rag_context_text="[source:brief.pdf chunk:rag-1]\nquarterly results",
    )

    assert citations == build_rag_citations(rag_chunks)


def test_select_chat_citations_ignores_rag_chunks_below_rerank_threshold(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "rerank_relevance_threshold", 0.1)
    rag_chunks = [
        {
            "source_title": "The-Founders-Playbook.pdf",
            "source_url": "",
            "chunk_id": "rag-low",
            "text": "irrelevant playbook chunk",
            "rerank_score": 0.008,
        },
        {
            "source_title": "brief.pdf",
            "source_url": "",
            "chunk_id": "rag-high",
            "text": "relevant brief chunk",
            "rerank_score": 0.42,
        },
    ]

    citations = select_chat_citations(
        rag_chunks,
        [],
        router_action="answer_from_rag",
        web_used=False,
        rag_context_text="[source:brief.pdf chunk:rag-high]\nrelevant brief chunk",
    )

    assert [citation["chunk_id"] for citation in citations] == ["rag-high"]


def test_select_chat_citations_does_not_fallback_after_irrelevant_rag_chunks(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "rerank_relevance_threshold", 0.1)

    citations = select_chat_citations(
        [
            {
                "source_title": "The-Founders-Playbook.pdf",
                "source_url": "",
                "chunk_id": "rag-low",
                "text": "irrelevant playbook chunk",
                "rerank_score": 0.008,
            }
        ],
        [],
        router_action="answer_from_rag",
        web_used=False,
        rag_context_text="[source:The-Founders-Playbook.pdf chunk:rag-low]\nirrelevant playbook chunk",
    )

    assert citations == []


def test_select_chat_citations_fallback_only_when_rag_context_exists():
    assert (
        select_chat_citations(
            [],
            [],
            router_action="answer_from_rag",
            web_used=False,
            rag_context_text="",
        )
        == []
    )

    citations = select_chat_citations(
        [],
        [],
        router_action="answer_from_rag",
        web_used=False,
        rag_context_text="[source:brief.pdf chunk:rag-1]\nquarterly results",
    )
    assert citations == [
        {
            "source_title": "workspace resources",
            "source_url": None,
            "chunk_id": "workspace-context-fallback",
            "text": "[source:brief.pdf chunk:rag-1]\nquarterly results",
            "source_type": "rag_fallback",
        }
    ]


def test_select_chat_citations_does_not_fallback_when_web_used_without_citations():
    citations = select_chat_citations(
        [
            {
                "source_title": "SaaS_Starter_Kit.pdf",
                "source_url": "",
                "chunk_id": "rag-1",
                "text": "starter kit",
            }
        ],
        [],
        router_action="web_search",
        web_used=True,
        rag_context_text="irrelevant workspace context",
    )
    assert citations == []


def test_select_chat_citations_excludes_rag_for_direct_answer():
    citations = select_chat_citations(
        [
            {
                "source_title": "playbook.pdf",
                "source_url": "",
                "chunk_id": "rag-1",
                "text": "Unrelated workspace content",
            }
        ],
        [],
        router_action="answer_direct",
        web_used=False,
        rag_context_text="Unrelated workspace content",
    )

    assert citations == []


def test_select_chat_citations_fails_closed_without_router_action():
    citations = select_chat_citations(
        [],
        [],
        router_action=None,
        web_used=False,
        rag_context_text="Workspace context without structured chunks",
    )

    assert citations == []
