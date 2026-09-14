"""Alert delivery: dashboard tone, in-call warning to the receiver, SMS (owner SK).

The verdict engine decides *that* a warning is due. This decides *how* it reaches
the person being called, per kind of call:

| Event | WebRTC harness / replay | Twilio call |
|---|---|---|
| ``beep`` | tone + banner on the receiver dashboard | tone + spoken caution played to the **receiver's leg only** (Conference participant announcement), plus the dashboard |
| ``sms`` | dashboard banner | spoken warning to the receiver, SMS to the receiver's number, dashboard |
| ``summary`` | dashboard | SMS summary after the call, dashboard |

The caller never hears the warning: an announcement to one conference participant
is played to that participant alone.

Twilio is reached over its REST API with ``httpx`` and HTTP basic auth, so no Twilio
SDK is needed. Without credentials the dispatcher runs **dry**: every alert it would
have sent is recorded and shown on the dashboard, so the whole flow can be rehearsed
before an account exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import quote, urlencode

from live_call.verdict_engine import Event, EventKind

API = "https://api.twilio.com/2010-04-01"


@dataclass(frozen=True)
class TwilioSettings:
    account_sid: str
    auth_token: str
    number: str  # the Twilio number, E.164
    receiver: str  # the protected person's phone, E.164
    public_base_url: str  # https URL Twilio can reach (ngrok)

    @property
    def configured(self) -> bool:
        placeholders = ("xxxx", "your-", "XXXX")
        values = (
            self.account_sid,
            self.auth_token,
            self.number,
            self.receiver,
            self.public_base_url,
        )
        return all(values) and not any(p in v for v in values for p in placeholders)

    @classmethod
    def from_env(cls) -> TwilioSettings:
        return cls(
            account_sid=os.environ.get("TWILIO_ACCOUNT_SID", ""),
            auth_token=os.environ.get("TWILIO_AUTH_TOKEN", ""),
            number=os.environ.get("TWILIO_NUMBER", ""),
            receiver=os.environ.get("RECEIVER_NUMBER", ""),
            public_base_url=os.environ.get("PUBLIC_BASE_URL", "").rstrip("/"),
        )


@dataclass
class SentAlert:
    session: str
    channel: str  # "announce" | "sms" | "dashboard"
    kind: str
    detail: dict
    dry_run: bool


class TwilioRest:
    """The four Twilio REST calls the demo needs."""

    def __init__(self, settings: TwilioSettings, client=None) -> None:
        self.settings = settings
        self._client = client

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                auth=(self.settings.account_sid, self.settings.auth_token), timeout=10.0
            )
        return self._client

    def _url(self, path: str) -> str:
        return f"{API}/Accounts/{self.settings.account_sid}/{path}"

    async def _post(self, path: str, data: dict) -> dict:
        response = await self._http().post(self._url(path), data=data)
        response.raise_for_status()
        return response.json()

    async def send_sms(self, body: str) -> dict:
        return await self._post(
            "Messages.json",
            {"From": self.settings.number, "To": self.settings.receiver, "Body": body},
        )

    async def call_receiver_into(self, conference: str) -> dict:
        """Dial the receiver and drop them into the named conference."""
        url = f"{self.settings.public_base_url}/twilio/join?{urlencode({'conference': conference})}"
        return await self._post(
            "Calls.json",
            {
                "From": self.settings.number,
                "To": self.settings.receiver,
                "Url": url,
                "StatusCallback": f"{self.settings.public_base_url}/twilio/status",
                "StatusCallbackEvent": "completed",
            },
        )

    async def conference_sid(self, friendly_name: str) -> str | None:
        response = await self._http().get(
            self._url("Conferences.json"),
            params={"FriendlyName": friendly_name, "Status": "in-progress"},
        )
        response.raise_for_status()
        conferences = response.json().get("conferences", [])
        return conferences[0]["sid"] if conferences else None

    async def announce_to_participant(self, conference_sid: str, call_sid: str, kind: str) -> dict:
        """Play a warning to one conference participant only."""
        url = f"{self.settings.public_base_url}/twilio/announce?kind={quote(kind)}"
        return await self._post(
            f"Conferences/{conference_sid}/Participants/{call_sid}.json",
            {"AnnounceUrl": url, "AnnounceMethod": "POST"},
        )


@dataclass
class AlertDispatcher:
    """Routes verdict events to the channels that exist for each call."""

    settings: TwilioSettings = field(default_factory=TwilioSettings.from_env)
    rest: TwilioRest | None = None
    sent: list[SentAlert] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.rest is None and self.settings.configured:
            self.rest = TwilioRest(self.settings)

    @property
    def live(self) -> bool:
        return self.rest is not None

    def _record(self, info, channel: str, kind: str, detail: dict, dry: bool) -> None:
        self.sent.append(SentAlert(info.session_id, channel, kind, detail, dry))
        del self.sent[:-200]

    async def deliver(self, info, event: Event) -> None:
        kind = event.kind.value
        self._record(info, "dashboard", kind, {"message": event.message}, dry=False)
        if info.kind != "twilio":
            return
        conference = info.meta.get("conference")
        receiver_call = info.meta.get("receiver_call_sid")
        dry = not self.live

        if event.kind in (EventKind.BEEP, EventKind.SMS) and conference:
            detail = {"conference": conference, "receiver_call_sid": receiver_call}
            if not receiver_call:
                detail["skipped"] = "receiver leg not connected yet"
            elif not dry:
                sid = await self.rest.conference_sid(conference)
                if sid:
                    await self.rest.announce_to_participant(sid, receiver_call, kind)
                    detail["conference_sid"] = sid
                else:
                    detail["skipped"] = "conference not in progress"
            self._record(info, "announce", kind, detail, dry)

        if event.kind in (EventKind.SMS, EventKind.SUMMARY):
            body = sms_body(event, info)
            if not dry:
                await self.rest.send_sms(body)
            self._record(info, "sms", kind, {"body": body}, dry)


def sms_body(event: Event, info) -> str:
    caller = info.meta.get("from", "the caller")
    if event.kind == EventKind.SMS:
        return (
            f"Deepfake warning: the voice of {caller} on your current call is likely "
            "cloned. Do not share OTPs, passwords or money. Verify by calling back."
        )
    return f"Call summary ({caller}): {event.message}"


ANNOUNCEMENTS = {
    "beep": "Caution. This caller's voice may be synthetic.",
    "sms": "Warning. This caller's voice is likely cloned. Do not share codes or money.",
}


def announcement_twiml(kind: str, beep_url: str | None = None) -> str:
    """TwiML for the receiver-only warning."""
    text = ANNOUNCEMENTS.get(kind, ANNOUNCEMENTS["beep"])
    play = f"<Play>{beep_url}</Play>" if beep_url else ""
    return f'<?xml version="1.0" encoding="UTF-8"?><Response>{play}<Say voice="alice">{text}</Say></Response>'
