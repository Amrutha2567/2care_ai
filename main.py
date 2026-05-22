"""
agent_core.py — Claude-powered agent with tool orchestration.

The agent:
1. Receives enriched context (patient history, session state, transcript)
2. Reasons over available tools (scheduling, memory)
3. Streams response tokens for low-latency TTS feeding
4. Logs full reasoning traces for observability

Tool calls are REAL — no hardcoded responses.
Reasoning traces are printed to stdout and logged to the session.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Callable, Any

import anthropic
import structlog

from memory.memory_manager import MemoryManager, SessionState, PatientContext
from scheduling.scheduling_service import SchedulingService, ConflictError, NotFoundError

log = structlog.get_logger()

# ── Tool Definitions ───────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "check_availability",
        "description": (
            "Check available appointment slots for a doctor or specialty. "
            "Use this when the patient asks about available times, wants to book, "
            "or needs to reschedule. Always call this before proposing a time to the patient."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doctor_id": {
                    "type": "string",
                    "description": "UUID of the doctor. Use if the patient specified a doctor."
                },
                "specialty": {
                    "type": "string",
                    "description": "Medical specialty (e.g. 'cardiology', 'dermatology'). Use if patient specifies specialty but not a specific doctor."
                },
                "date_str": {
                    "type": "string",
                    "description": "Date to check availability for, in YYYY-MM-DD format."
                },
                "days_ahead": {
                    "type": "integer",
                    "description": "Number of days ahead to search. Default 7.",
                    "default": 7
                }
            },
            "required": []
        }
    },
    {
        "name": "book_appointment",
        "description": (
            "Book an appointment for the patient. Only call this AFTER the patient has "
            "explicitly confirmed they want to book the specific slot. "
            "Always confirm the details verbally before calling this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "slot_id": {
                    "type": "string",
                    "description": "UUID of the availability slot to book."
                },
                "reason": {
                    "type": "string",
                    "description": "Patient's stated reason for the visit (optional)."
                }
            },
            "required": ["slot_id"]
        }
    },
    {
        "name": "reschedule_appointment",
        "description": (
            "Reschedule an existing appointment to a new slot. "
            "First use get_patient_appointments to find the appointment_id, "
            "then check_availability to find the new slot_id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "appointment_id": {
                    "type": "string",
                    "description": "UUID of the appointment to reschedule."
                },
                "new_slot_id": {
                    "type": "string",
                    "description": "UUID of the new slot."
                }
            },
            "required": ["appointment_id", "new_slot_id"]
        }
    },
    {
        "name": "cancel_appointment",
        "description": (
            "Cancel an existing appointment. Only call this after the patient "
            "has explicitly confirmed they want to cancel. "
            "Offer to reschedule first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "appointment_id": {
                    "type": "string",
                    "description": "UUID of the appointment to cancel."
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for cancellation (optional)."
                }
            },
            "required": ["appointment_id"]
        }
    },
    {
        "name": "get_patient_appointments",
        "description": (
            "Retrieve the patient's upcoming and/or past appointments. "
            "Use this when the patient asks about their bookings or when you need "
            "an appointment_id to reschedule or cancel."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "include_past": {
                    "type": "boolean",
                    "description": "Whether to include past appointments. Default false.",
                    "default": False
                }
            },
            "required": []
        }
    },
    {
        "name": "find_alternatives",
        "description": (
            "Find alternative appointment slots when the patient's preferred slot or time is unavailable. "
            "Use this after a conflict is detected or when the patient says a proposed time doesn't work."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doctor_id": {
                    "type": "string",
                    "description": "Doctor UUID if patient wants same doctor."
                },
                "specialty": {
                    "type": "string",
                    "description": "Specialty if patient is flexible on doctor."
                },
                "preferred_date_str": {
                    "type": "string",
                    "description": "Starting date to search from, YYYY-MM-DD."
                },
                "count": {
                    "type": "integer",
                    "description": "Number of alternatives to return. Default 3.",
                    "default": 3
                }
            },
            "required": []
        }
    },
    {
        "name": "update_language_preference",
        "description": (
            "Update the patient's preferred language. Call this if the patient "
            "switches language mid-call and confirms they prefer that language."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["en", "hi", "ta"],
                    "description": "Language code: en (English), hi (Hindi), ta (Tamil)."
                }
            },
            "required": ["language"]
        }
    }
]


# ── System Prompt ──────────────────────────────────────────────────────────

def build_system_prompt(
    patient_context: Optional[PatientContext],
    language: str,
    is_outbound: bool = False,
    campaign_script: Optional[str] = None,
) -> str:
    lang_instructions = {
        "en": "Respond in English. Be warm, professional, and concise.",
        "hi": (
            "Respond in Hindi. Use simple, conversational Hindi. "
            "Avoid overly formal language. Medical terms may remain in English "
            "if no clear Hindi equivalent exists (e.g., 'appointment', 'doctor'). "
            "Use आप (formal you) when addressing the patient."
        ),
        "ta": (
            "Respond in Tamil. Use polite second-person forms (நீங்கள்). "
            "Medical terms in English are acceptable. Be warm and respectful."
        ),
    }

    patient_section = ""
    if patient_context:
        patient_section = f"""
## Patient Information
{patient_context.to_prompt_fragment()}
"""

    outbound_section = ""
    if is_outbound and campaign_script:
        outbound_section = f"""
## Outbound Call Context
This is an OUTBOUND call initiated by the clinic. You are calling the patient, not receiving their call.
Start with the campaign script below, then handle their response naturally.

Campaign Script:
{campaign_script}
"""

    return f"""You are VoiceRx, a clinical appointment assistant for a healthcare platform.
You handle appointment booking, rescheduling, cancellation, and general queries through voice.

## Language
{lang_instructions.get(language, lang_instructions["en"])}
{patient_section}{outbound_section}
## Behaviour Guidelines

**Confirmation pattern:** Always verbally confirm details before executing any write operation (book/cancel/reschedule). Say what you're about to do and ask the patient to confirm with "yes" or "haan" or "aamaam" etc.

**Conflict handling:** If a slot is unavailable, immediately offer alternatives. Never leave the patient without options.

**Brevity:** Voice responses should be 1-3 sentences maximum. Do not list more than 3 options at once. Avoid filler phrases.

**Clarity on codes:** When giving confirmation codes, read each character clearly: "Your confirmation code is Alpha-Foxtrot-Seven-Two" or in Hindi spell it out in Hindi.

**Uncertainty:** If you don't understand something, ask one specific clarifying question. Do not guess.

**Closing:** After completing an action, ask if there's anything else. If not, close warmly in 1 sentence.

**Do not:**
- Invent slot times not returned by tools
- Book without explicit patient confirmation
- Disclose other patients' information
- Provide medical advice or diagnoses

## Tool Usage
Use tools in this order for booking:
1. check_availability → present 2-3 options → patient confirms → book_appointment
2. For rescheduling: get_patient_appointments → check_availability → reschedule_appointment
3. For cancellation: get_patient_appointments → confirm with patient → cancel_appointment

Always call tools with real parameters. Never fabricate tool results.
"""


# ── Latency Trace ─────────────────────────────────────────────────────────

@dataclass
class LatencyTrace:
    call_sid: str
    turn: int
    stt_end_ms: float = 0
    lang_detect_ms: float = 0
    redis_read_ms: float = 0
    first_llm_token_ms: float = 0
    sentence_boundary_ms: float = 0
    tts_first_chunk_ms: float = 0
    total_ms: float = 0
    timestamps: dict = field(default_factory=dict)

    def record(self, stage: str):
        self.timestamps[stage] = time.perf_counter() * 1000

    def compute_total(self, start_ms: float):
        now = time.perf_counter() * 1000
        self.total_ms = now - start_ms
        return self


# ── Agent ─────────────────────────────────────────────────────────────────

class VoiceAgent:
    """
    Orchestrates the LLM + tool calls for a single conversation turn.
    Streams response text for low-latency TTS feeding.
    """

    def __init__(
        self,
        anthropic_client: anthropic.AsyncAnthropic,
        scheduling_service: SchedulingService,
        memory_manager: MemoryManager,
        db_session_factory: Callable,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.client = anthropic_client
        self.scheduling = scheduling_service
        self.memory = memory_manager
        self.db_factory = db_session_factory
        self.model = model

    async def process_turn(
        self,
        transcript: str,
        session: SessionState,
        patient_context: Optional[PatientContext],
        on_token: Callable[[str], None],
        trace: Optional[LatencyTrace] = None,
    ) -> str:
        """
        Process a single conversation turn.

        Args:
            transcript: The patient's spoken input (transcribed).
            session: Current session state from Redis.
            patient_context: Patient history from PostgreSQL.
            on_token: Callback called with each streamed text token.
            trace: Latency trace object to populate.

        Returns:
            The complete assistant response text.
        """
        if trace:
            trace.record("agent_start")

        # Add user turn to history
        session.add_turn("user", transcript)

        # Build messages
        messages = session.to_claude_messages()

        # Build system prompt
        system = build_system_prompt(
            patient_context=patient_context,
            language=session.language,
            is_outbound=session.is_outbound,
        )

        full_response = ""
        tool_calls_this_turn = []

        # Agentic loop — keeps calling tools until a final text response
        iteration = 0
        max_iterations = 5  # Safety bound

        while iteration < max_iterations:
            iteration += 1
            first_token = True

            log.info(
                "agent.llm_call",
                call_sid=session.call_sid,
                turn=session.turn_count,
                iteration=iteration,
                message_count=len(messages),
            )

            async with self.client.messages.stream(
                model=self.model,
                max_tokens=512,
                system=system,
                messages=messages,
                tools=TOOLS,
            ) as stream:
                current_tool_use = None
                current_tool_input_json = ""
                response_text = ""
                stop_reason = None

                async for event in stream:
                    event_type = type(event).__name__

                    # Track first token latency
                    if first_token and hasattr(event, "type"):
                        if event.type in ("content_block_start", "content_block_delta"):
                            first_token = False
                            if trace:
                                trace.record("first_llm_token")

                    # Stream text tokens immediately to TTS
                    if hasattr(event, "type") and event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "type"):
                            if delta.type == "text_delta":
                                token = delta.text
                                response_text += token
                                on_token(token)
                            elif delta.type == "input_json_delta":
                                current_tool_input_json += delta.partial_json

                    # Detect tool use block start
                    if hasattr(event, "type") and event.type == "content_block_start":
                        block = event.content_block
                        if hasattr(block, "type") and block.type == "tool_use":
                            current_tool_use = {
                                "id": block.id,
                                "name": block.name,
                            }
                            current_tool_input_json = ""

                    # Tool use block complete
                    if hasattr(event, "type") and event.type == "content_block_stop":
                        if current_tool_use and current_tool_input_json:
                            try:
                                tool_input = json.loads(current_tool_input_json)
                            except json.JSONDecodeError:
                                tool_input = {}
                            current_tool_use["input"] = tool_input
                            tool_calls_this_turn.append(current_tool_use.copy())
                            current_tool_use = None
                            current_tool_input_json = ""

                    # Capture stop reason
                    if hasattr(event, "type") and event.type == "message_delta":
                        if hasattr(event.delta, "stop_reason"):
                            stop_reason = event.delta.stop_reason

            # Log reasoning trace
            if tool_calls_this_turn:
                for tc in tool_calls_this_turn:
                    log.info(
                        "agent.tool_call",
                        call_sid=session.call_sid,
                        tool=tc["name"],
                        input=tc.get("input", {}),
                    )

            # If no tool calls, we have a final text response
            if not tool_calls_this_turn or stop_reason == "end_turn":
                full_response = response_text
                session.add_turn("assistant", full_response)
                break

            # Execute tool calls and append results
            tool_results = []
            for tool_call in tool_calls_this_turn:
                result = await self._execute_tool(
                    tool_name=tool_call["name"],
                    tool_input=tool_call.get("input", {}),
                    session=session,
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": json.dumps(result),
                })
                log.info(
                    "agent.tool_result",
                    call_sid=session.call_sid,
                    tool=tool_call["name"],
                    result_keys=list(result.keys()) if isinstance(result, dict) else "n/a",
                )

            # Append assistant turn with tool calls + tool results for next iteration
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response_text} if response_text else None,
                    *[
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc.get("input", {}),
                        }
                        for tc in tool_calls_this_turn
                    ]
                ],
            })
            messages[-1]["content"] = [c for c in messages[-1]["content"] if c]

            messages.append({"role": "user", "content": tool_results})
            tool_calls_this_turn = []

        if trace:
            trace.record("agent_end")

        return full_response

    async def _execute_tool(
        self, tool_name: str, tool_input: dict, session: SessionState
    ) -> dict:
        """Dispatch tool calls to the scheduling service."""
        patient_id = session.patient_id

        async with self.db_factory() as db:
            try:
                if tool_name == "check_availability":
                    return await self.scheduling.check_availability(db=db, **tool_input)

                elif tool_name == "book_appointment":
                    if not patient_id:
                        return {"error": "Patient not identified. Cannot book appointment."}
                    return await self.scheduling.book_appointment(
                        db=db,
                        patient_id=patient_id,
                        session_id=session.call_sid,
                        **tool_input,
                    )

                elif tool_name == "reschedule_appointment":
                    if not patient_id:
                        return {"error": "Patient not identified."}
                    return await self.scheduling.reschedule_appointment(
                        db=db,
                        patient_id=patient_id,
                        **tool_input,
                    )

                elif tool_name == "cancel_appointment":
                    if not patient_id:
                        return {"error": "Patient not identified."}
                    return await self.scheduling.cancel_appointment(
                        db=db,
                        patient_id=patient_id,
                        **tool_input,
                    )

                elif tool_name == "get_patient_appointments":
                    if not patient_id:
                        return {"error": "Patient not identified."}
                    return await self.scheduling.get_patient_appointments(
                        db=db,
                        patient_id=patient_id,
                        **tool_input,
                    )

                elif tool_name == "find_alternatives":
                    return await self.scheduling.find_alternatives(db=db, **tool_input)

                elif tool_name == "update_language_preference":
                    lang = tool_input.get("language", session.language)
                    session.language = lang
                    session.entities_extracted["language_updated"] = lang
                    return {"success": True, "language": lang}

                else:
                    return {"error": f"Unknown tool: {tool_name}"}

            except ConflictError as e:
                log.warning("tool.conflict", tool=tool_name, error=str(e))
                return {"error": str(e), "conflict": True}

            except NotFoundError as e:
                log.warning("tool.not_found", tool=tool_name, error=str(e))
                return {"error": str(e), "not_found": True}

            except Exception as e:
                log.error("tool.unexpected_error", tool=tool_name, error=str(e))
                return {"error": "An unexpected error occurred. Please try again."}

    async def generate_session_summary(self, session: SessionState) -> str:
        """
        Generate a concise summary of the session for cross-session memory.
        Uses a lightweight call (not streaming).
        """
        if not session.conversation_history:
            return "No meaningful conversation recorded."

        history_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}"
            for t in session.conversation_history[-10:]
        )

        message = await self.client.messages.create(
            model="claude-haiku-4-5-20251001",  # Haiku for cheap summarisation
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Summarise this patient call in 2-3 sentences. "
                        f"Note: language used, what they asked about, and the outcome.\n\n"
                        f"{history_text}"
                    ),
                }
            ],
        )
        return message.content[0].text if message.content else "Session summary unavailable."
