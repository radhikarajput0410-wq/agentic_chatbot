from backend import (
    chatbot,
    get_all_threads,
    get_last_human_message,
    delete_thread,
    register_thread,
    ingest_rag_document,
    set_conversation_title,
    get_conversation_title,
)

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

from langgraph.types import Command
from streamlit_cookies_controller import CookieController

import streamlit as st
import uuid
import tempfile
import os


cookies = CookieController()

_SESSION_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365


def generate_thread_id():
    return str(uuid.uuid4())


def init_session_and_thread():
    """
    Restore BOTH the visitor's identity (session_id) and the exact
    conversation they had open (thread_id) from cookies in a single
    round-trip.

    Why thread_id needs its own cookie too: st.session_state is wiped
    completely on a hard browser refresh -- only cookies survive that.
    Previously only session_id was cookied, so on every refresh
    thread_id fell back to `generate_thread_id()` and silently pointed
    the UI at a brand-new, empty conversation. The old conversation was
    never actually deleted (it's still in the sidebar), but the active
    chat window looked wiped -- that was the actual bug. Storing the
    active thread_id in its own cookie (kept in sync every time the
    user starts a new chat or switches conversations -- see
    reset_chat/switch_to_thread) fixes that: a refresh now reopens the
    exact conversation the user was just looking at.
    """
    if "session_id" in st.session_state and "thread_id" in st.session_state:
        return st.session_state["session_id"], st.session_state["thread_id"]

    all_cookies = cookies.getAll()

    if all_cookies is None:
        st.stop()

    existing_sid = all_cookies.get("sid")
    existing_tid = all_cookies.get("tid")

    session_id = existing_sid or str(uuid.uuid4())
    thread_id = existing_tid or generate_thread_id()

    needs_rerun = False

    if not existing_sid:
        cookies.set(
            "sid",
            session_id,
            max_age=_SESSION_COOKIE_MAX_AGE_SECONDS,
        )
        needs_rerun = True

    if not existing_tid:
        cookies.set(
            "tid",
            thread_id,
            max_age=_SESSION_COOKIE_MAX_AGE_SECONDS,
        )
        needs_rerun = True

    st.session_state["session_id"] = session_id
    st.session_state["thread_id"] = thread_id

    if needs_rerun:
        # Force a rerun so both cookie writes are confirmed/available
        # before the rest of the script relies on them.
        st.rerun()

    return session_id, thread_id


def set_current_thread_cookie(thread_id):
    """
    Keep the "tid" cookie in sync with whichever conversation is
    currently active, so a refresh always restores THIS conversation,
    not whatever was active when the cookie was first written.
    """
    cookies.set(
        "tid",
        thread_id,
        max_age=_SESSION_COOKIE_MAX_AGE_SECONDS,
    )


def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, thread_id)

    register_thread(thread_id, st.session_state["session_id"])


# ========================= Chat title helpers =========================

def get_thread_title(thread_id):
    """
    Get a short, human-readable title for a conversation thread.

    Precedence:
      1. A custom title the user explicitly set via rename (persisted
         in the conversation_meta table -- survives refresh/restart).
      2. A title derived from the thread's first user message.
      3. "New Chat" for empty/new conversations.

    Titles are cached in session state so we don't re-hit the
    checkpointer/database on every rerun.
    """

    titles = st.session_state.setdefault("thread_titles", {})

    if thread_id in titles:
        return titles[thread_id]

    custom_title = get_conversation_title(thread_id)

    if custom_title:
        titles[thread_id] = custom_title
        return custom_title

    first_message_text = get_last_human_message(thread_id)
    title = _shorten_title(first_message_text) if first_message_text else "New Chat"

    titles[thread_id] = title
    return title


def _shorten_title(text):
    """Trim a message down to a short sidebar-friendly title."""

    cleaned = " ".join(text.strip().split())

    if len(cleaned) > 30:
        cleaned = cleaned[:30].rstrip() + "..."

    return cleaned or "New Chat"


def cache_thread_title(thread_id, text):
    """
    Immediately cache a title for a thread from the text just sent,
    so the sidebar can update without waiting for another checkpoint
    read. Only sets it if the thread doesn't already have a real title
    (a renamed/custom title always wins and is never overwritten here).
    """

    titles = st.session_state.setdefault("thread_titles", {})

    if get_conversation_title(thread_id):
        # User already renamed this conversation -- never clobber that.
        return

    if titles.get(thread_id, "New Chat") == "New Chat":
        titles[thread_id] = _shorten_title(text)


def rename_thread(thread_id, new_title):
    """
    Persist a user-chosen title for a conversation and update the
    in-memory cache immediately so the sidebar reflects it without
    waiting for another rerun cycle.
    """

    new_title = (new_title or "").strip()

    if not new_title:
        return

    set_conversation_title(thread_id, new_title)

    titles = st.session_state.setdefault("thread_titles", {})
    titles[thread_id] = new_title


def remove_thread(thread_id):
    """
    Delete a conversation thread everywhere: the database, the
    sidebar list, and the cached title.
    """

    delete_thread(thread_id)

    if thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(thread_id)

    st.session_state.get("thread_titles", {}).pop(thread_id, None)

    # Also clear any in-progress rename UI for this thread
    if st.session_state.get("renaming_thread_id") == thread_id:
        st.session_state["renaming_thread_id"] = None


# Create a completely new chat conversation
def reset_chat():

    st.session_state["thread_id"] = generate_thread_id()

    # Keep the "tid" cookie in sync so a refresh reopens THIS new
    # chat, not the previous one.
    set_current_thread_cookie(st.session_state["thread_id"])

    st.session_state["message_history"] = []

    st.session_state["pending_hitl"] = None

    add_thread(st.session_state["thread_id"])


# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


def _convert_messages_for_display(messages):
    """
    Convert LangChain messages (as stored in the LangGraph checkpoint)
    into the plain dict format the Streamlit UI renders. Shared by
    switch_to_thread() (clicking a sidebar conversation) AND the
    startup/refresh path -- a refresh needs exactly the same
    reconstruction, not just an empty message_history. Defined here
    (before it's first used at module level, right after
    load_conversation) rather than down by switch_to_thread, since
    Streamlit executes this file top-to-bottom as a script every rerun.
    """

    temp_messages = []

    pending_image_path = None

    for message in messages:

        if isinstance(message, ToolMessage):

            if getattr(message, "name", None) == "generate_image":
                tool_content = message.content or ""

                if tool_content.startswith("IMAGE_FILE::"):
                    pending_image_path = tool_content.split("IMAGE_FILE::", 1)[1]

            continue

        if isinstance(message, HumanMessage):
            role = "user"

        elif isinstance(message, AIMessage):
            role = "assistant"

        else:
            continue

        entry = {
            "role": role,
            "content": message.content
        }

        if role == "assistant" and pending_image_path:
            entry["image_path"] = pending_image_path
            pending_image_path = None

        temp_messages.append(entry)

    return temp_messages


# ========================= HITL helper functions =========================

def get_pending_interrupt(thread_id):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        state_snapshot = chatbot.get_state(config)

        direct_interrupts = getattr(
            state_snapshot,
            "interrupts",
            ()
        ) or ()

        if direct_interrupts:
            return direct_interrupts[0]

        tasks = getattr(
            state_snapshot,
            "tasks",
            ()
        ) or ()

        for task in tasks:

            task_interrupts = getattr(
                task,
                "interrupts",
                ()
            ) or ()

            if task_interrupts:
                return task_interrupts[0]

    except Exception:

        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):

    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):

    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:

        save_pending_interrupt(
            thread_id,
            pending_interrupt
        )

    else:

        current_pending = st.session_state.get(
            "pending_hitl"
        )

        if (
            current_pending is not None
            and current_pending.get("thread_id") == thread_id
        ):
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):

    pending_hitl = st.session_state.get(
        "pending_hitl"
    )

    if not pending_hitl:

        st.warning(
            "There is no pending action to approve or reject."
        )

        return

    interrupted_thread_id = pending_hitl["thread_id"]

    resume_config = {
        "configurable": {
            "thread_id": interrupted_thread_id,
            "session_id": st.session_state["session_id"]
        },
        "metadata": {
            "thread_id": interrupted_thread_id
        },
        "run_name": "hitl_resume_trace",
    }

    try:

        with st.chat_message("assistant"):

            status_holder = {
                "box": st.status(
                    "🔄 Resuming the requested action...",
                    expanded=True
                )
            }

            def resumed_ai_only_stream():

                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):

                    if isinstance(
                        message_chunk,
                        ToolMessage
                    ):

                        tool_name = getattr(
                            message_chunk,
                            "name",
                            "tool"
                        )

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                    if (
                        isinstance(message_chunk, AIMessage)
                        and metadata.get("langgraph_node") == "chat_node"
                    ):

                        if message_chunk.content:
                            yield message_chunk.content

            resumed_ai_message = st.write_stream(
                resumed_ai_only_stream()
            )

            next_interrupt = get_pending_interrupt(
                interrupted_thread_id
            )

            if next_interrupt is not None:

                save_pending_interrupt(
                    interrupted_thread_id,
                    next_interrupt
                )

                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False
                )

            else:

                st.session_state["pending_hitl"] = None

                status_holder["box"].update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False
                )

        if resumed_ai_message:

            st.session_state["message_history"].append({
                "role": "assistant",
                "content": resumed_ai_message
            })

        st.rerun()

    except Exception as error:

        st.error(
            f"Could not resume the requested action: {error}"
        )


# ========================= Page configuration =========================

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖"
)

st.title("CHATTERBOX")


init_session_and_thread()


if "message_history" not in st.session_state:
    # Reload this thread's actual saved messages from the LangGraph
    # checkpointer -- NOT an empty list. thread_id at this point is
    # either a restored one (from the "tid" cookie, meaning it may
    # already have messages) or a genuinely new one (load_conversation
    # simply returns [] for that case). Either way this is what
    # actually makes chat history survive a refresh.
    st.session_state["message_history"] = _convert_messages_for_display(
        load_conversation(st.session_state["thread_id"])
    )


if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads(st.session_state["session_id"])


if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None


# Tracks which thread (if any) currently has its rename text-input open
if "renaming_thread_id" not in st.session_state:
    st.session_state["renaming_thread_id"] = None


add_thread(st.session_state["thread_id"])


sync_pending_interrupt(
    st.session_state["thread_id"]
)


# ========================= Sidebar threading feature =========================

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] button {
        text-align: left;
        justify-content: flex-start;
        border-radius: 8px;
        border: 1px solid transparent;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] button:hover {
        border: 1px solid rgba(255, 255, 255, 0.25);
        background-color: rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] button[kind="primary"] {
        background-color: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("My Conversations")


if st.sidebar.button("➕ New Chat", use_container_width=True):

    reset_chat()

    st.rerun()

st.sidebar.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)


def switch_to_thread(thread_id):
    """Load a conversation into the active chat window."""

    st.session_state["thread_id"] = thread_id

    # Keep the "tid" cookie in sync so a refresh reopens THIS
    # conversation, not whichever one was active before the switch.
    set_current_thread_cookie(thread_id)

    messages = load_conversation(thread_id)

    st.session_state["message_history"] = _convert_messages_for_display(messages)

    sync_pending_interrupt(thread_id)

    st.rerun()


# Display all conversation threads, newest first, with a
# ChatGPT/Claude-style readable title, a rename option, and a
# delete option per chat
for thread_id in st.session_state["chat_threads"]:

    title = get_thread_title(thread_id)
    is_active = thread_id == st.session_state["thread_id"]
    is_renaming = st.session_state["renaming_thread_id"] == thread_id

    if is_renaming:

        # Renaming UI replaces the normal row for just this thread
        rename_col, save_col, cancel_col = st.sidebar.columns([4, 1, 1])

        with rename_col:
            new_title_input = st.text_input(
                "Rename conversation",
                value=title,
                key=f"rename_input_{thread_id}",
                label_visibility="collapsed",
            )

        with save_col:
            if st.button("✅", key=f"save_rename_{thread_id}", help="Save name"):
                rename_thread(thread_id, new_title_input)
                st.session_state["renaming_thread_id"] = None
                st.rerun()

        with cancel_col:
            if st.button("✖️", key=f"cancel_rename_{thread_id}", help="Cancel"):
                st.session_state["renaming_thread_id"] = None
                st.rerun()

    else:

        title_col, rename_trigger_col, delete_col = st.sidebar.columns([4, 1, 1])

        with title_col:
            if st.button(
                title,
                key=f"open_{thread_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                switch_to_thread(thread_id)

        with rename_trigger_col:
            if st.button(
                "✏️",
                key=f"rename_{thread_id}",
                use_container_width=True,
                help="Rename this conversation"
            ):
                st.session_state["renaming_thread_id"] = thread_id
                st.rerun()

        with delete_col:
            if st.button(
                "🗑️",
                key=f"delete_{thread_id}",
                use_container_width=True,
                help="Delete this conversation"
            ):
                remove_thread(thread_id)

                if is_active:
                    reset_chat()

                st.rerun()


# ========================= Main chat interface =========================

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        image_path = message.get("image_path")
        if image_path and os.path.exists(image_path):
            st.image(image_path)


# ========================= HITL approval interface =========================

pending_hitl = st.session_state.get(
    "pending_hitl"
)

current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id")
    == st.session_state["thread_id"]
)


if current_thread_has_pending_hitl:

    st.warning(
        "🧑 Human approval required\n\n"
        f"{pending_hitl['prompt']}"
    )

    approve_column, reject_column = st.columns(2)

    with approve_column:

        if st.button(
            "✅ Approve Purchase",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            use_container_width=True
        ):

            resume_hitl_execution("yes")

    with reject_column:

        if st.button(
            "❌ Reject Purchase",
            key=f"reject_{st.session_state['thread_id']}",
            use_container_width=True
        ):

            resume_hitl_execution("no")


# ========================= Fixed chat input with PDF upload =========================

submission = st.chat_input(
    "Type here",
    accept_file=True,
    file_type=["pdf"],

    disabled=current_thread_has_pending_hitl
)


user_input = None


if submission:

    user_input = submission.text

    uploaded_files = submission.files

    if uploaded_files:

        uploaded_pdf = uploaded_files[0]

        temporary_file_path = None

        raw_pdf_bytes = uploaded_pdf.getvalue()

        looks_like_a_real_pdf = (
            len(raw_pdf_bytes) >= 1024
            and b"%PDF-" in raw_pdf_bytes[:1024]
        )

        if not looks_like_a_real_pdf:

            st.error(
                f"\"{uploaded_pdf.name}\" was received incomplete "
                f"({len(raw_pdf_bytes)} bytes) and couldn't be processed. "
                "This usually happens when a file is picked directly from "
                "Google Drive before it has fully downloaded to the "
                "device. Try opening the file once in the Drive app first "
                "(or downloading it), then upload it here again -- or pick "
                "it from Files/Downloads (local storage) instead, which "
                "doesn't have this issue."
            )

        else:

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temporary_file:

                    temporary_file.write(raw_pdf_bytes)

                    temporary_file_path = temporary_file.name

                with st.spinner(
                    f"Processing {uploaded_pdf.name}..."
                ):

                    ingest_rag_document(
                        temporary_file_path,
                        session_id=st.session_state["session_id"],
                        thread_id=st.session_state["thread_id"],
                        filename=uploaded_pdf.name
                    )

                st.toast(
                    f"{uploaded_pdf.name} processed successfully.",
                    icon="✅"
                )

            except Exception as error:

                st.error(
                    f"PDF processing failed: {error}"
                )

            finally:

                if (
                    temporary_file_path
                    and os.path.exists(temporary_file_path)
                ):
                    os.remove(temporary_file_path)


if user_input:

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    cache_thread_title(st.session_state["thread_id"], user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"],
            "session_id": st.session_state["session_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }

    with st.chat_message("assistant"):

        status_holder = {
            "box": None
        }

        generated_image_holder = {
            "path": None
        }

        def ai_only_stream():

            yielded_any_content = False

            try:
                for message_chunk, metadata in chatbot.stream(
                    {
                        "messages": [
                            HumanMessage(content=user_input)
                        ]
                    },
                    config=CONFIG,
                    stream_mode="messages",
                ):

                    if isinstance(
                        message_chunk,
                        ToolMessage
                    ):

                        tool_name = getattr(
                            message_chunk,
                            "name",
                            "tool"
                        )

                        if tool_name == "generate_image":

                            tool_content = message_chunk.content or ""

                            if tool_content.startswith("IMAGE_FILE::"):
                                generated_image_holder["path"] = (
                                    tool_content.split("IMAGE_FILE::", 1)[1]
                                )

                        if status_holder["box"] is None:

                            status_holder["box"] = st.status(
                                f"🔧 Using `{tool_name}` …",
                                expanded=True
                            )

                        else:

                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` …",
                                state="running",
                                expanded=True,
                            )

                    if (
                        isinstance(message_chunk, AIMessage)
                        and metadata.get("langgraph_node") == "chat_node"
                    ):
                        yielded_any_content = True
                        yield message_chunk.content

            except Exception:
                yield (
                    "Sorry, something went wrong while generating a "
                    "response. Please try again, or rephrase your question."
                )
                return

            if not yielded_any_content:

                final_state = chatbot.get_state(config=CONFIG)
                final_messages = final_state.values.get("messages", [])

                if final_messages and isinstance(final_messages[-1], AIMessage):
                    fallback_content = final_messages[-1].content

                    if fallback_content:
                        yield fallback_content

            pending_interrupt = get_pending_interrupt(
                st.session_state["thread_id"]
            )

            if pending_interrupt is not None:

                save_pending_interrupt(
                    st.session_state["thread_id"],
                    pending_interrupt
                )

                yield (
                    "\n\n⚠️ This stock purchase requires your approval. "
                    "Use the Approve Purchase or Reject Purchase "
                    "button below."
                )

        ai_message = st.write_stream(
            ai_only_stream()
        )

        if generated_image_holder["path"] and os.path.exists(generated_image_holder["path"]):
            st.image(generated_image_holder["path"])

        if status_holder["box"] is not None:

            if get_pending_interrupt(
                st.session_state["thread_id"]
            ) is not None:

                status_holder["box"].update(
                    label="⏸️ Waiting for human approval",
                    state="complete",
                    expanded=False
                )

            else:

                status_holder["box"].update(
                    label="✅ Tool finished",
                    state="complete",
                    expanded=False
                )

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message,
        "image_path": generated_image_holder["path"]
    })

    if (
        st.session_state.get("pending_hitl") is not None
        and st.session_state["pending_hitl"].get("thread_id")
        == st.session_state["thread_id"]
    ):
        st.rerun()

    st.rerun()