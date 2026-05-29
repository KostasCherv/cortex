from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.api.endpoints import app
from src.auth import AuthenticatedUser, get_authenticated_user
from src.itinerary import (
    ItineraryPlannerResponse,
    ItineraryPlannerValidationError,
    ItinerarySessionDetail,
    ItinerarySessionMessage,
    ItinerarySessionListResponse,
    ItinerarySessionSummary,
    ItineraryVersion,
    PlannerTravelRequirements,
)


def _auth_override() -> AuthenticatedUser:
    return AuthenticatedUser(user_id="test-user", email="test@example.com")


app.dependency_overrides[get_authenticated_user] = _auth_override
client = TestClient(app)


def _session_summary(**overrides) -> ItinerarySessionSummary:
    payload = {
        "session_id": "itin-1",
        "owner_id": "test-user",
        "workspace_id": "test-user",
        "title": "Paris spring city break",
        "status": "generated",
        "current_version_id": "ver-1",
        "prompt_preview": "Plan a 4 day Paris trip",
        "last_message_preview": "Make it cheaper",
        "created_at": "2026-05-29T10:00:00+00:00",
        "updated_at": "2026-05-29T10:00:00+00:00",
    }
    payload.update(overrides)
    return ItinerarySessionSummary(**payload)


def _session_detail(**overrides) -> ItinerarySessionDetail:
    version = ItineraryVersion(
        version_id="ver-1",
        session_id="itin-1",
        version_number=1,
        revision_summary="Initial itinerary",
        markdown="# Paris spring city break",
        itinerary={
            "title": "Paris spring city break",
            "summary": "A relaxed four-day trip.",
            "destination": "Paris",
            "budget_band": "mid-range",
            "days": [],
            "tips": [],
        },
        created_at="2026-05-29T10:00:00+00:00",
    )
    return ItinerarySessionDetail(
        session_id="itin-1",
        owner_id="test-user",
        workspace_id="test-user",
        title="Paris spring city break",
        status="generated",
        requirements=PlannerTravelRequirements(
            destination="Paris",
            start_date="2026-06-10",
            end_date="2026-06-14",
            trip_length_days=4,
            traveler_count=2,
            party_type="couple",
            budget_band="mid-range",
            interests=["art", "cafes"],
            constraints=["avoid rushed mornings"],
            pace="relaxed",
        ),
        current_version_id="ver-1",
        prompt_preview="Plan a 4 day Paris trip",
        last_message_preview="Make it cheaper",
        created_at="2026-05-29T10:00:00+00:00",
        updated_at="2026-05-29T10:00:00+00:00",
        messages=[
            ItinerarySessionMessage(
                message_id="msg-1",
                session_id="itin-1",
                role="assistant",
                content="Tell me where you want to go.",
                metadata={},
                created_at="2026-05-29T10:00:00+00:00",
            )
        ],
        versions=[version],
        current_version=version,
        **overrides,
    )


def _planner_response(**overrides) -> ItineraryPlannerResponse:
    detail = _session_detail(**overrides.pop("session_overrides", {}))
    assistant_message = ItinerarySessionMessage(
        message_id="msg-2",
        session_id="itin-1",
        role="assistant",
        content="I updated your itinerary.",
        metadata={"action": "revise"},
        created_at="2026-05-29T10:05:00+00:00",
    )
    return ItineraryPlannerResponse(
        session=detail,
        assistant_message=assistant_message,
        current_itinerary=detail.current_version.structured_itinerary if detail.current_version else None,
        new_version=detail.current_version,
        created_new_version=True,
        missing_fields=[],
        **overrides,
    )


def test_create_itinerary_session():
    created = _session_summary(status="collecting_requirements", current_version_id=None)
    with patch("src.api.endpoints.create_itinerary_session", new=AsyncMock(return_value=created)) as create_mock:
        response = client.post("/api/itinerary/sessions", json={})

    assert response.status_code == 200
    assert response.json()["session_id"] == "itin-1"
    create_mock.assert_awaited_once_with("test-user")


def test_list_itinerary_sessions():
    with patch(
        "src.api.endpoints.list_itinerary_sessions",
        new=AsyncMock(return_value=ItinerarySessionListResponse(sessions=[_session_summary()])),
    ) as list_mock:
        response = client.get("/api/itinerary/sessions")

    assert response.status_code == 200
    assert response.json()["sessions"][0]["session_id"] == "itin-1"
    list_mock.assert_awaited_once_with("test-user")


def test_get_itinerary_session_detail():
    with patch("src.api.endpoints.get_itinerary_session_detail", new=AsyncMock(return_value=_session_detail())) as get_mock:
        response = client.get("/api/itinerary/sessions/itin-1")

    assert response.status_code == 200
    assert response.json()["session_id"] == "itin-1"
    get_mock.assert_awaited_once_with("itin-1", "test-user")


def test_process_itinerary_message():
    response_payload = _planner_response()
    with patch(
        "src.api.endpoints.process_itinerary_session_message",
        new=AsyncMock(return_value=response_payload),
    ) as process_mock:
        response = client.post("/api/itinerary/sessions/itin-1/messages", json={"message": "Make it cheaper"})

    assert response.status_code == 200
    assert response.json()["assistant_message"]["content"] == "I updated your itinerary."
    process_mock.assert_awaited_once_with("itin-1", "test-user", "Make it cheaper")


def test_process_itinerary_message_maps_validation_errors():
    with patch(
        "src.api.endpoints.process_itinerary_session_message",
        new=AsyncMock(
            side_effect=ItineraryPlannerValidationError(
                "itinerary_generation_failed",
                "Generated itinerary output could not be validated.",
            )
        ),
    ):
        response = client.post("/api/itinerary/sessions/itin-1/messages", json={"message": "Make it cheaper"})

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "itinerary_generation_failed"
