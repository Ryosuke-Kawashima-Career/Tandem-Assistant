"""
Summary:
    test_agent_tools_calendar.py is the executable specification for REQ-20: the agent
    schedules a follow-up session on Google Calendar and invites the participants, over
    server-side credentials, announcing the result rather than failing silently.

Covers:
    - REQ-20 event construction: start/end from a duration, attendees, invite delivery.
    - REQ-20 credential gate and vendor failure handling.
    - REQ-20 dispatch: `meeting.scheduled` on success, `tool.status` otherwise.
    - REQ-20's email rule: Resend supplements a real invitation, and never replaces one.
"""

import unittest
from unittest.mock import MagicMock

from src.agent.tools.base import (
    ToolInvocationError,
    ToolNotConfiguredError,
    ToolState,
)
from src.agent.tools.dispatch import ToolDispatcher
from src.agent.tools.email import ResendEmailSender
from src.agent.tools.google_calendar import GoogleCalendarTool
from src.sessions.models import SessionRecord
from src.server import app, server_instance

TOKEN = "test-calendar-token"
START = "2026-09-10T09:00:00+00:00"

CALENDAR_RESPONSE = {
    "id": "evt-abc123",
    "htmlLink": "https://calendar.google.com/event?eid=evt-abc123",
    "status": "confirmed",
}


def work_session():
    """An international-work session whose follow-up is being booked."""
    return SessionRecord.create(
        channel="calendar-channel", mode="international_work",
        languages=["en"], participants=["Priya", "Kenji"]
    )


class RecordingTransport:
    """A stand-in for the HTTP POST to a vendor API."""

    def __init__(self, response=None, error=None):
        """Store the response to replay, or the error to raise instead."""
        self.response = response if response is not None else CALENDAR_RESPONSE
        self.error = error
        self.calls = []

    def __call__(self, url, payload, headers=None):
        """Record the request and replay the canned outcome."""
        self.calls.append({"url": url, "payload": payload, "headers": headers or {}})
        if self.error is not None:
            raise self.error
        return self.response


class TestGoogleCalendarTool(unittest.TestCase):
    """Test suite for the Calendar client itself (REQ-20, TASK-12.4)."""

    def test_a_tool_without_a_token_is_unconfigured(self):
        """
        Verify no credential means no call.

        REQ-20 keeps Calendar credentials server-side; a request built without one
        returns a vendor 401 that says nothing useful to the person who clicked
        "schedule".
        """
        tool = GoogleCalendarTool(access_token="", credentials_json="")

        self.assertFalse(tool.is_configured)
        with self.assertRaises(ToolNotConfiguredError):
            tool.schedule(summary="Follow-up", start_time=START)

    def test_scheduling_posts_an_event_with_attendees_and_invitations(self):
        """
        Verify the event carries its attendees and asks Google to deliver the invites.

        `sendUpdates=all` is the difference between an event in the organizer's own
        calendar and an invitation the participants actually receive.
        """
        transport = RecordingTransport()
        tool = GoogleCalendarTool(access_token=TOKEN, transport=transport)

        result = tool.schedule(
            summary="EchoSphere follow-up",
            start_time=START,
            duration_minutes=45,
            attendees=["priya@example.com", "kenji@example.com"],
            description="Continue the migration discussion."
        )

        self.assertEqual(result["event_id"], "evt-abc123")
        self.assertEqual(result["html_link"], CALENDAR_RESPONSE["htmlLink"])
        self.assertEqual(result["attendees"], ["priya@example.com", "kenji@example.com"])

        call = transport.calls[0]
        self.assertIn("sendUpdates=all", call["url"])
        self.assertEqual(call["headers"]["Authorization"], f"Bearer {TOKEN}")
        self.assertEqual(call["payload"]["summary"], "EchoSphere follow-up")
        self.assertEqual(
            [entry["email"] for entry in call["payload"]["attendees"]],
            ["priya@example.com", "kenji@example.com"]
        )

    def test_the_end_time_follows_from_the_duration(self):
        """Verify a 45-minute meeting ends 45 minutes after it starts."""
        transport = RecordingTransport()
        tool = GoogleCalendarTool(access_token=TOKEN, transport=transport)

        result = tool.schedule(summary="Follow-up", start_time=START, duration_minutes=45)

        self.assertEqual(result["start_time"], "2026-09-10T09:00:00+00:00")
        self.assertEqual(result["end_time"], "2026-09-10T09:45:00+00:00")

    def test_an_unusable_attendee_address_is_dropped_not_sent(self):
        """
        Verify a malformed address does not take the whole invitation down with it.

        Google rejects the entire request over one bad attendee, which would turn a
        typo in one participant's name into no meeting for anybody.
        """
        transport = RecordingTransport()
        tool = GoogleCalendarTool(access_token=TOKEN, transport=transport)

        result = tool.schedule(
            summary="Follow-up", start_time=START,
            attendees=["priya@example.com", "Kenji", ""]
        )

        self.assertEqual(result["attendees"], ["priya@example.com"])
        self.assertEqual(len(transport.calls[0]["payload"]["attendees"]), 1)

    def test_a_vendor_failure_becomes_a_tool_invocation_error(self):
        """Verify a rejected request surfaces as this module's own error type."""
        tool = GoogleCalendarTool(
            access_token=TOKEN, transport=RecordingTransport(error=RuntimeError("403"))
        )

        with self.assertRaises(ToolInvocationError):
            tool.schedule(summary="Follow-up", start_time=START)

    def test_a_token_provider_supplies_the_credential_when_no_token_is_set(self):
        """
        Verify a service account can mint the token instead of an env-pinned one.

        A static `GOOGLE_CALENDAR_ACCESS_TOKEN` expires within the hour, so a deployment
        that runs longer than a demo needs the provider path to exist.
        """
        transport = RecordingTransport()
        tool = GoogleCalendarTool(
            transport=transport, token_provider=lambda: "minted-token"
        )

        self.assertTrue(tool.is_configured)
        tool.schedule(summary="Follow-up", start_time=START)

        self.assertEqual(transport.calls[0]["headers"]["Authorization"], "Bearer minted-token")


class TestCalendarDispatch(unittest.TestCase):
    """Test suite for announcing a scheduled meeting (REQ-20, TASK-12.4)."""

    def setUp(self):
        """A recording data stream, a configured Calendar tool, and a Resend sender."""
        self.data_stream = MagicMock()
        self.data_stream.send_tool_event.return_value = True
        self.calendar_transport = RecordingTransport()
        self.email_transport = RecordingTransport(response={"id": "email-1"})
        self.dispatcher = ToolDispatcher(
            data_stream=self.data_stream,
            calendar=GoogleCalendarTool(
                access_token=TOKEN, transport=self.calendar_transport
            ),
            email=ResendEmailSender(
                api_key="re_test", sender="tandem@example.com",
                transport=self.email_transport
            )
        )

    def published(self, event_type):
        """Returns the payloads published under one event type."""
        return [
            call.args[1] for call in self.data_stream.send_tool_event.call_args_list
            if call.args[0] == event_type
        ]

    def test_a_scheduled_meeting_is_announced_over_rtc(self):
        """Verify `meeting.scheduled` carries the event id, start time, and attendees."""
        session = work_session()

        result = self.dispatcher.schedule_meeting(
            session, summary="Follow-up", start_time=START,
            attendees=["priya@example.com"]
        )

        self.assertTrue(result.ok)
        payload = self.published("meeting.scheduled")[0]
        self.assertEqual(payload["session_id"], session.session_id)
        self.assertEqual(payload["meeting"]["event_id"], "evt-abc123")
        self.assertEqual(payload["meeting"]["start_time"], START)
        self.assertEqual(payload["meeting"]["attendees"], ["priya@example.com"])

    def test_a_confirmation_email_supplements_a_real_invitation(self):
        """
        Verify Resend is used only after Calendar accepted the event (REQ-20).

        The invitation itself is Google's; the email is a courtesy copy. Sending it
        instead of an invitation would leave a "meeting" that exists in nobody's
        calendar.
        """
        self.dispatcher.schedule_meeting(
            work_session(), summary="Follow-up", start_time=START,
            attendees=["priya@example.com"]
        )

        self.assertEqual(len(self.calendar_transport.calls), 1)
        self.assertEqual(len(self.email_transport.calls), 1)
        self.assertIn("priya@example.com", str(self.email_transport.calls[0]["payload"]["to"]))

    def test_a_failed_confirmation_email_does_not_fail_the_meeting(self):
        """
        Verify a bounced courtesy email leaves the booked meeting booked.

        The event already exists in everyone's calendar by this point; reporting failure
        would invite a second, duplicate booking.
        """
        dispatcher = ToolDispatcher(
            data_stream=self.data_stream,
            calendar=GoogleCalendarTool(access_token=TOKEN, transport=RecordingTransport()),
            email=ResendEmailSender(
                api_key="re_test", sender="tandem@example.com",
                transport=RecordingTransport(error=RuntimeError("domain not verified"))
            )
        )

        result = dispatcher.schedule_meeting(
            work_session(), summary="Follow-up", start_time=START,
            attendees=["priya@example.com"]
        )

        self.assertTrue(result.ok)
        self.assertFalse(result.payload["confirmation_email_sent"])

    def test_no_email_is_sent_when_the_calendar_call_failed(self):
        """Verify a failed booking never produces an email announcing it."""
        dispatcher = ToolDispatcher(
            data_stream=self.data_stream,
            calendar=GoogleCalendarTool(
                access_token=TOKEN, transport=RecordingTransport(error=RuntimeError("403"))
            ),
            email=ResendEmailSender(
                api_key="re_test", sender="tandem@example.com",
                transport=self.email_transport
            )
        )

        result = dispatcher.schedule_meeting(
            work_session(), summary="Follow-up", start_time=START,
            attendees=["priya@example.com"]
        )

        self.assertEqual(result.state, ToolState.FAILED)
        self.assertEqual(self.email_transport.calls, [])
        self.assertEqual(self.published("tool.status")[0]["state"], "failed")

    def test_an_unconfigured_calendar_reports_unavailable_not_silence(self):
        """
        Verify a missing credential is announced (REQ-20 forbids a silent no-op).

        Someone who clicked "schedule" and saw nothing happen will assume the meeting
        exists.
        """
        dispatcher = ToolDispatcher(
            data_stream=self.data_stream, calendar=GoogleCalendarTool(access_token="")
        )

        result = dispatcher.schedule_meeting(
            work_session(), summary="Follow-up", start_time=START
        )

        self.assertEqual(result.state, ToolState.UNAVAILABLE)
        status = self.published("tool.status")[0]
        self.assertEqual(status["tool"], "calendar")
        self.assertTrue(status["reason"])
        self.assertFalse(self.published("meeting.scheduled"))


class TestCalendarScheduleEndpoint(unittest.TestCase):
    """Test suite for the scheduling endpoint (REQ-20, REQ-16)."""

    def setUp(self):
        """Start from an empty registry with the server's Calendar tool captured."""
        self.app = app.test_client()
        self.server = server_instance
        self.server.sessions.reset()
        self.server.artifacts.reset()
        self.original_calendar = self.server.tools.calendar
        self.app.post("/api/session/start", json={
            "channel": "calendar-channel", "mode": "international_work",
            "participants": ["Priya"], "languages": ["en"]
        })

    def tearDown(self):
        """Restore the server's own tool."""
        self.server.tools.calendar = self.original_calendar

    def test_scheduling_returns_the_event_and_its_link(self):
        """Verify a configured server books the meeting and reports where it is."""
        self.server.tools.calendar = GoogleCalendarTool(
            access_token=TOKEN, transport=RecordingTransport()
        )

        response = self.app.post("/api/tools/calendar/schedule", json={
            "channel": "calendar-channel", "actor": "Priya",
            "summary": "Follow-up", "start_time": START,
            "duration_minutes": 30, "attendees": ["priya@example.com"]
        })

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertEqual(body["meeting"]["event_id"], "evt-abc123")

    def test_a_missing_start_time_is_a_400(self):
        """Verify the endpoint validates before spending a vendor call."""
        self.server.tools.calendar = GoogleCalendarTool(
            access_token=TOKEN, transport=RecordingTransport()
        )

        response = self.app.post("/api/tools/calendar/schedule", json={
            "channel": "calendar-channel", "actor": "Priya", "summary": "Follow-up"
        })

        self.assertEqual(response.status_code, 400)

    def test_an_unconfigured_calendar_answers_503(self):
        """Verify an unconfigured integration refuses plainly."""
        self.server.tools.calendar = GoogleCalendarTool(access_token="")

        response = self.app.post("/api/tools/calendar/schedule", json={
            "channel": "calendar-channel", "actor": "Priya",
            "summary": "Follow-up", "start_time": START
        })

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["state"], "unavailable")

    def test_a_non_participant_may_not_schedule_on_a_session(self):
        """Verify scheduling is governed: attendee lists are participant data (REQ-16)."""
        self.server.tools.calendar = GoogleCalendarTool(
            access_token=TOKEN, transport=RecordingTransport()
        )

        response = self.app.post("/api/tools/calendar/schedule", json={
            "channel": "calendar-channel", "actor": "Stranger",
            "summary": "Follow-up", "start_time": START
        })

        self.assertEqual(response.status_code, 403)


class TestToolStatusEndpoint(unittest.TestCase):
    """Test suite for reporting which tools this server actually has (REQ-18–20)."""

    def setUp(self):
        """A test client over the shared server instance."""
        self.app = app.test_client()

    def test_the_status_endpoint_reports_each_tool_without_leaking_credentials(self):
        """
        Verify the UI can tell which controls to offer, and learns nothing else.

        The frontend needs a per-tool boolean to avoid offering a button that can only
        produce a 503; it has no business knowing the key behind it.
        """
        response = self.app.get("/api/tools/status")

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertEqual(
            set(body["tools"]), {"search", "anki", "calendar", "email"}
        )
        for configured in body["tools"].values():
            self.assertIsInstance(configured, bool)


if __name__ == "__main__":
    unittest.main()
