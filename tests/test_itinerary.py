import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.itinerary import (
    ItineraryPlannerResponse,
    ItineraryPlannerValidationError,
    ItinerarySessionDetail,
    ItinerarySessionMessage,
    ItinerarySessionSummary,
    ItineraryVersion,
    PlannerTravelRequirements,
    process_itinerary_message,
)


class _LLMResult:
    def __init__(self, content: str) -> None:
        self.content = content


def _json_result(payload: object) -> _LLMResult:
    return _LLMResult(json.dumps(payload))


def _session_detail(
    *,
    current_version_id: str | None = None,
    title: str = "Paris spring city break",
    status: str = "collecting_requirements",
    requirements: PlannerTravelRequirements | None = None,
    messages: list[ItinerarySessionMessage] | None = None,
    versions: list[ItineraryVersion] | None = None,
) -> ItinerarySessionDetail:
    return ItinerarySessionDetail(
        session_id="itin-1",
        owner_id="test-user",
        workspace_id="test-user",
        title=title,
        status=status,
        requirements=requirements or PlannerTravelRequirements(),
        current_version_id=current_version_id,
        prompt_preview="Plan a Paris trip",
        last_message_preview="Need a 4 day trip in Paris.",
        created_at="2026-05-29T10:00:00+00:00",
        updated_at="2026-05-29T10:00:00+00:00",
        messages=messages or [],
        versions=versions or [],
        current_version=(versions or [None])[-1] if versions else None,
    )


def _persisted_message(
    *,
    role: str,
    content: str,
    metadata: dict | None = None,
    message_id: str = "msg-generated",
) -> ItinerarySessionMessage:
    return ItinerarySessionMessage(
        message_id=message_id,
        session_id="itin-1",
        role=role,  # type: ignore[arg-type]
        content=content,
        metadata=metadata or {},
        created_at="2026-05-29T10:05:00+00:00",
    )


def _persisted_version(
    *,
    version_number: int,
    revision_summary: str = "Generated itinerary draft.",
    itinerary: dict | None = None,
    version_id: str | None = None,
) -> ItineraryVersion:
    return ItineraryVersion(
        version_id=version_id or f"ver-{version_number}",
        session_id="itin-1",
        version_number=version_number,
        revision_summary=revision_summary,
        markdown="# Saved itinerary\n",
        itinerary=itinerary
        or {
            "title": "Saved itinerary",
            "summary": "Saved summary",
            "destination": "Paris",
            "budget_band": "mid-range",
            "days": [],
            "tips": [],
        },
        created_at="2026-05-29T10:05:00+00:00",
    )


@pytest.mark.asyncio
async def test_process_itinerary_message_collects_requirements_and_asks_follow_up():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        _json_result(
            {
                "destination": "Paris",
                "trip_length_days": 4,
                "traveler_count": 2,
                "party_type": "couple",
                "budget_band": "mid-range",
                "interests": ["art", "cafes"],
            }
        ),
        _LLMResult("What dates are you considering for the trip?"),
    ]

    session = _session_detail()

    with (
        patch("src.llm.factory.get_llm", return_value=mock_llm),
        patch(
            "src.itinerary.append_itinerary_message",
            new=AsyncMock(
                side_effect=[
                    _persisted_message(
                        role="user",
                        content="I want a 4 day Paris trip for two people with a mid-range budget and lots of art and cafes.",
                        message_id="msg-user-1",
                    ),
                    _persisted_message(
                        role="assistant",
                        content="What dates are you considering for the trip?",
                        metadata={"action": "collect_requirements"},
                        message_id="msg-assistant-1",
                    ),
                ]
            ),
        ) as append_message_mock,
        patch("src.itinerary.update_itinerary_session", new=AsyncMock()) as update_session_mock,
    ):
        response = await process_itinerary_message(
            session=session,
            user_id="test-user",
            message="I want a 4 day Paris trip for two people with a mid-range budget and lots of art and cafes.",
        )

    assert isinstance(response, ItineraryPlannerResponse)
    assert response.session.status == "collecting_requirements"
    assert response.session.requirements.destination == "Paris"
    assert response.session.requirements.trip_length_days == 4
    assert response.session.requirements.interests == ["art", "cafes"]
    assert response.new_version is None
    assert "dates" in response.assistant_message.content.lower()
    assert append_message_mock.await_count == 2
    update_session_mock.assert_awaited()


@pytest.mark.asyncio
async def test_process_itinerary_message_generates_first_itinerary_and_version():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        _json_result(
            {
                "destination": "Paris",
                "start_date": "2026-06-10",
                "end_date": "2026-06-14",
                "trip_length_days": 4,
                "traveler_count": 2,
                "party_type": "couple",
                "budget_band": "mid-range",
                "interests": ["art", "cafes"],
                "constraints": ["avoid rushed mornings"],
                "pace": "relaxed",
            }
        ),
        _json_result(
            {
                "title": "Paris art and cafe getaway",
                "summary": "A relaxed four-day Paris itinerary with museums, neighborhoods, and cafe stops.",
                "destination": "Paris",
                "budget_band": "mid-range",
                "days": [
                    {
                        "day_number": 1,
                        "title": "Arrival and Left Bank",
                        "morning": ["Check in and slow breakfast"],
                        "afternoon": ["Musee d'Orsay"],
                        "evening": ["Seine walk and dinner"],
                        "notes": ["Keep the first day light"],
                    }
                ],
                "tips": ["Reserve the museum in advance"],
            }
        ),
    ]

    requirements = PlannerTravelRequirements(
        destination="Paris",
        traveler_count=2,
        party_type="couple",
        budget_band="mid-range",
        interests=["art", "cafes"],
        pace="relaxed",
    )
    session = _session_detail(requirements=requirements)

    with (
        patch("src.llm.factory.get_llm", return_value=mock_llm),
        patch(
            "src.itinerary.append_itinerary_message",
            new=AsyncMock(
                side_effect=[
                    _persisted_message(
                        role="user",
                        content="We want to go from June 10 to June 14 and avoid rushed mornings.",
                        message_id="msg-user-2",
                    ),
                    _persisted_message(
                        role="assistant",
                        content="I generated your itinerary. A relaxed four-day Paris itinerary with museums, neighborhoods, and cafe stops.",
                        metadata={"action": "generate_itinerary"},
                        message_id="msg-assistant-2",
                    ),
                ]
            ),
        ) as append_message_mock,
        patch("src.itinerary.update_itinerary_session", new=AsyncMock()) as update_session_mock,
        patch(
            "src.itinerary.create_itinerary_version",
            new=AsyncMock(
                return_value=_persisted_version(
                    version_number=1,
                    itinerary={
                        "title": "Paris art and cafe getaway",
                        "summary": "A relaxed four-day Paris itinerary with museums, neighborhoods, and cafe stops.",
                        "destination": "Paris",
                        "budget_band": "mid-range",
                        "days": [
                            {
                                "day_number": 1,
                                "title": "Arrival and Left Bank",
                                "morning": ["Check in and slow breakfast"],
                                "afternoon": ["Musee d'Orsay"],
                                "evening": ["Seine walk and dinner"],
                                "notes": ["Keep the first day light"],
                            }
                        ],
                        "tips": ["Reserve the museum in advance"],
                    },
                    revision_summary="Generated the first itinerary draft.",
                    version_id="ver-1",
                )
            ),
        ) as create_version_mock,
        patch("src.itinerary.search_destination_context", new=AsyncMock(return_value="Paris travel notes")),
    ):
        response = await process_itinerary_message(
            session=session,
            user_id="test-user",
            message="We want to go from June 10 to June 14 and avoid rushed mornings.",
        )

    assert response.session.status == "generated"
    assert response.new_version is not None
    assert response.new_version.version_number == 1
    assert response.current_itinerary is not None
    assert response.current_itinerary.title == "Paris art and cafe getaway"
    assert "generated" in response.assistant_message.content.lower()
    assert append_message_mock.await_count == 2
    update_session_mock.assert_awaited()
    create_version_mock.assert_awaited()


@pytest.mark.asyncio
async def test_process_itinerary_message_revises_existing_itinerary_and_adds_new_version():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _json_result(
        {
            "title": "Paris art and cafe getaway",
            "summary": "A revised itinerary with a lighter budget and Montmartre on day two.",
            "destination": "Paris",
            "budget_band": "budget-conscious",
            "days": [
                {
                    "day_number": 1,
                    "title": "Arrival and Left Bank",
                    "morning": ["Pastry breakfast"],
                    "afternoon": ["Walk the Latin Quarter"],
                    "evening": ["Budget bistro dinner"],
                    "notes": ["Use metro instead of taxis"],
                }
            ],
            "tips": ["Swap one museum for a neighborhood walk"],
            "revision_summary": "Reduced the spend and added more neighborhood time.",
        }
    )

    version = ItineraryVersion(
        version_id="ver-1",
        session_id="itin-1",
        version_number=1,
        revision_summary="Initial itinerary",
        markdown="# Paris art and cafe getaway",
        itinerary={
            "title": "Paris art and cafe getaway",
            "summary": "Initial itinerary",
            "destination": "Paris",
            "budget_band": "mid-range",
            "days": [],
            "tips": [],
        },
        created_at="2026-05-29T10:00:00+00:00",
    )
    session = _session_detail(
        current_version_id="ver-1",
        status="generated",
        versions=[version],
    )

    with (
        patch("src.llm.factory.get_llm", return_value=mock_llm),
        patch(
            "src.itinerary.append_itinerary_message",
            new=AsyncMock(
                side_effect=[
                    _persisted_message(
                        role="user",
                        content="Make it cheaper and add more neighborhood time.",
                        message_id="msg-user-3",
                    ),
                    _persisted_message(
                        role="assistant",
                        content="I updated your itinerary. Reduced the spend and added more neighborhood time.",
                        metadata={"action": "revise_itinerary"},
                        message_id="msg-assistant-3",
                    ),
                ]
            ),
        ) as append_message_mock,
        patch("src.itinerary.update_itinerary_session", new=AsyncMock()) as update_session_mock,
        patch(
            "src.itinerary.create_itinerary_version",
            new=AsyncMock(
                return_value=_persisted_version(
                    version_number=2,
                    revision_summary="Reduced the spend and added more neighborhood time.",
                    itinerary={
                        "title": "Paris art and cafe getaway",
                        "summary": "A revised itinerary with a lighter budget and Montmartre on day two.",
                        "destination": "Paris",
                        "budget_band": "budget-conscious",
                        "days": [
                            {
                                "day_number": 1,
                                "title": "Arrival and Left Bank",
                                "morning": ["Pastry breakfast"],
                                "afternoon": ["Walk the Latin Quarter"],
                                "evening": ["Budget bistro dinner"],
                                "notes": ["Use metro instead of taxis"],
                            }
                        ],
                        "tips": ["Swap one museum for a neighborhood walk"],
                    },
                    version_id="ver-2",
                )
            ),
        ) as create_version_mock,
        patch("src.itinerary.search_destination_context", new=AsyncMock(return_value="Paris travel notes")),
    ):
        response = await process_itinerary_message(
            session=session,
            user_id="test-user",
            message="Make it cheaper and add more neighborhood time.",
        )

    assert response.session.status == "generated"
    assert response.new_version is not None
    assert response.new_version.version_number == 2
    assert response.current_itinerary is not None
    assert response.current_itinerary.budget_band == "budget-conscious"
    assert "updated" in response.assistant_message.content.lower()
    assert append_message_mock.await_count == 2
    update_session_mock.assert_awaited()
    create_version_mock.assert_awaited()


@pytest.mark.asyncio
async def test_process_itinerary_message_raises_validation_error_when_generation_response_is_invalid():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        _json_result(
            {
                "destination": "Paris",
                "start_date": "2026-06-10",
                "end_date": "2026-06-14",
                "trip_length_days": 4,
                "traveler_count": 2,
                "party_type": "couple",
                "budget_band": "mid-range",
                "interests": ["art", "cafes"],
                "constraints": ["avoid rushed mornings"],
                "pace": "relaxed",
            }
        ),
        _LLMResult("not valid json"),
        _LLMResult("still not valid json"),
    ]

    session = _session_detail(
        requirements=PlannerTravelRequirements(
            destination="Paris",
            traveler_count=2,
            party_type="couple",
            budget_band="mid-range",
            interests=["art", "cafes"],
            pace="relaxed",
        )
    )

    with (
        patch("src.llm.factory.get_llm", return_value=mock_llm),
        patch(
            "src.itinerary.append_itinerary_message",
            new=AsyncMock(
                return_value=_persisted_message(
                    role="user",
                    content="We want to go from June 10 to June 14 and avoid rushed mornings.",
                    message_id="msg-user-4",
                )
            ),
        ),
        patch("src.itinerary.search_destination_context", new=AsyncMock(return_value="Paris travel notes")),
    ):
        with pytest.raises(ItineraryPlannerValidationError) as exc_info:
            await process_itinerary_message(
                session=session,
                user_id="test-user",
                message="We want to go from June 10 to June 14 and avoid rushed mornings.",
            )

    assert exc_info.value.code == "itinerary_generation_failed"
