"""Tests for citation provenance and final citation selection."""

from src.api.deps import (
    _build_rag_citations,
    _build_web_citations,
    _select_chat_citations,
)


def test_build_rag_citations_includes_source_type():
    citations = _build_rag_citations(
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
    citations = _build_web_citations(
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

    citations = _select_chat_citations(
        rag_chunks,
        web_citations,
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

    citations = _select_chat_citations(
        rag_chunks,
        [],
        web_used=False,
        rag_context_text="[source:brief.pdf chunk:rag-1]\nquarterly results",
    )

    assert citations == _build_rag_citations(rag_chunks)


def test_select_chat_citations_fallback_only_when_rag_context_exists():
    assert _select_chat_citations([], [], web_used=False, rag_context_text="") == []

    citations = _select_chat_citations(
        [],
        [],
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
    citations = _select_chat_citations(
        [
            {
                "source_title": "SaaS_Starter_Kit.pdf",
                "source_url": "",
                "chunk_id": "rag-1",
                "text": "starter kit",
            }
        ],
        [],
        web_used=True,
        rag_context_text="irrelevant workspace context",
    )
    assert citations == []
