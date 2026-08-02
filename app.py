from backend import (
    chatbot,
    get_all_threads,
    get_last_human_message,
    delete_thread,
    register_thread,
    ingest_rag_document
)

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

from langgraph.types import Command

import streamlit as st
import uuid
import tempfile
import os


# Generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())


# ========================= Multi-user session isolation =========================
# Every browser tab gets its own session_id. It's stored in the page's URL
# query params (not just st.session_state) so a page REFRESH keeps the same
# session_id -- st.session_state alone would not survive a hard reload.
# Opening the app fresh (no "sid" in the URL) always creates a brand new,
# independent session, e.g. a different browser or device.
def init_session_id():
    if "session_id" in st.session_state:
        return st.session_state["session_id"]

    existing_sid = st.query_params.get("sid")

    if existing_sid:
        session_id = existing_sid
    else:
        session_id = str(uuid.uuid4())
        st.query_params["sid"] = session_id

    st.session_state["session_id"] = session_id
    return session_id


# Add a new thread ID to the conversation list
def add_thread(thread_id):

    # Prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:

        # Insert at the front so the newest conversation is always
        # shown first in the sidebar, like ChatGPT / Claude
        st.session_state["chat_threads"].insert(0, thread_id)

    # Scope this thread to the current browser session so no other
    # visitor's sidebar or history can ever include it
    register_thread(thread_id, st.session_state["session_id"])


# ========================= Chat title helpers =========================

def get_thread_title(thread_id):
    """
    Get a short, human-readable title for a conversation thread,
    derived from its first user message. Falls back to "New Chat"
    for empty conversations. Titles are cached in session state so
    we don't re-hit the checkpointer on every rerun.
    """

    titles = st.session_state.setdefault("thread_titles", {})

    if thread_id in titles:
        return titles[thread_id]

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
    read. Only sets it if the thread doesn't already have a real title.
    """

    titles = st.session_state.setdefault("thread_titles", {})

    if titles.get(thread_id, "New Chat") == "New Chat":
        titles[thread_id] = _shorten_title(text)


def remove_thread(thread_id):
    """
    Delete a conversation thread everywhere: the database, the
    sidebar list, and the cached title.
    """

    delete_thread(thread_id)

    if thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(thread_id)

    st.session_state.get("thread_titles", {}).pop(thread_id, None)


# Create a completely new chat conversation
def reset_chat():

    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # ========================= HITL ADDED =========================
    # Clear any pending human approval request
    st.session_state["pending_hitl"] = None
    # =============================================================

    # Add the new thread to the conversation list
    add_thread(st.session_state["thread_id"])


# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])


# ========================= HITL helper functions =========================

def get_pending_interrupt(thread_id):
    """
    Return the first unresolved LangGraph interrupt for a thread.

    Returns:
        The pending Interrupt object, or None.
    """

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        # Read the current checkpoint state
        state_snapshot = chatbot.get_state(config)

        # Some LangGraph versions expose interrupts directly
        direct_interrupts = getattr(
            state_snapshot,
            "interrupts",
            ()
        ) or ()

        if direct_interrupts:
            return direct_interrupts[0]

        # Other LangGraph versions store interrupts inside tasks
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

        # A newly created thread may not have a checkpoint yet
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    """
    Save the pending interrupt information inside Streamlit state.
    """

    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):
    """
    Synchronize Streamlit HITL state with the LangGraph checkpoint.

    This allows a pending approval request to reappear after:
    - a Streamlit rerun
    - a browser refresh
    - switching between conversations
    """

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
    """
    Resume an interrupted LangGraph execution.

    Args:
        decision:
            "yes" approves the stock purchase.
            "no" rejects the stock purchase.
    """

    pending_hitl = st.session_state.get(
        "pending_hitl"
    )

    if not pending_hitl:

        st.warning(
            "There is no pending action to approve or reject."
        )

        return

    # Get the thread that originally triggered the interrupt
    interrupted_thread_id = pending_hitl["thread_id"]

    # The same thread ID must be used when resuming
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

        # Display the resumed response
        with st.chat_message("assistant"):

            status_holder = {
                "box": st.status(
                    "🔄 Resuming the requested action...",
                    expanded=True
                )
            }

            def resumed_ai_only_stream():

                # Resume the graph with the human decision
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):

                    # Update tool execution status
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

                    # Stream only assistant-generated text that came
                    # from chat_node itself (see ai_only_stream above
                    # for why this filter matters)
                    if (
                        isinstance(message_chunk, AIMessage)
                        and metadata.get("langgraph_node") == "chat_node"
                    ):

                        if message_chunk.content:
                            yield message_chunk.content

            # Display the streamed final answer
            resumed_ai_message = st.write_stream(
                resumed_ai_only_stream()
            )

            # Check whether another interrupt occurred
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

                # No more pending approval
                st.session_state["pending_hitl"] = None

                status_holder["box"].update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False
                )

        # Save the assistant response in Streamlit UI history
        if resumed_ai_message:

            st.session_state["message_history"].append({
                "role": "assistant",
                "content": resumed_ai_message
            })

        # Rerun so the response appears in normal chat order
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

# Display the main application title
st.title("Agentic Chatbot with LangGraph")


# Establish this visitor's session_id before anything thread-related,
# since every thread lookup/creation below is scoped to it
init_session_id()


# Create message_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Create a list for storing all conversation thread IDs, scoped to this
# visitor's session only -- never another visitor's conversations
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads(st.session_state["session_id"])


# ========================= HITL ADDED =========================

# Store the currently pending human approval request
if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

# =============================================================


# Add the current thread to the conversation list
add_thread(st.session_state["thread_id"])


# ========================= HITL ADDED =========================

# Recover pending approval after page refresh or rerun
sync_pending_interrupt(
    st.session_state["thread_id"]
)

# =============================================================


# ========================= Sidebar threading feature =========================

# Style the sidebar so it reads like a ChatGPT / Claude conversation list
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

# Display the sidebar title
st.sidebar.title("My Conversations")


# Create a button for starting a new conversation
if st.sidebar.button("➕ New Chat", use_container_width=True):

    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()

st.sidebar.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)


def switch_to_thread(thread_id):
    """Load a conversation into the active chat window."""

    # Set the selected thread as the current thread
    st.session_state["thread_id"] = thread_id

    # Load the messages saved under the selected thread
    messages = load_conversation(thread_id)

    # Temporary list for converting LangChain messages
    # into Streamlit's required message format
    temp_messages = []

    # Tracks an image path seen from a generate_image ToolMessage until
    # the next assistant message, so it can be attached to that message
    # the same way the live streaming path does (see generated_image_holder
    # in the main input-handling block below).
    pending_image_path = None

    # Loop through all saved messages
    for message in messages:

        # ToolMessages aren't displayed directly, but a generate_image
        # result needs to be captured so its image can be re-attached to
        # the assistant message that follows it.
        if isinstance(message, ToolMessage):

            if getattr(message, "name", None) == "generate_image":
                tool_content = message.content or ""

                if tool_content.startswith("IMAGE_FILE::"):
                    pending_image_path = tool_content.split("IMAGE_FILE::", 1)[1]

            continue

        # Check whether the message was sent by the user
        if isinstance(message, HumanMessage):
            role = "user"

        # Check whether the message was sent by the AI
        elif isinstance(message, AIMessage):
            role = "assistant"

        # Ignore other message types
        else:
            continue

        # Convert the LangChain message into a dictionary
        entry = {
            "role": role,
            "content": message.content
        }

        if role == "assistant" and pending_image_path:
            entry["image_path"] = pending_image_path
            pending_image_path = None

        temp_messages.append(entry)

    # Replace the current UI history with the selected conversation
    st.session_state["message_history"] = temp_messages

    # ========================= HITL ADDED =========================

    # Restore any pending approval for this conversation
    sync_pending_interrupt(thread_id)

    # =============================================================

    # Rerun the application to display the loaded messages
    st.rerun()


# Display all conversation threads, newest first, with a
# ChatGPT/Claude-style readable title and a delete option per chat
for thread_id in st.session_state["chat_threads"]:

    title = get_thread_title(thread_id)
    is_active = thread_id == st.session_state["thread_id"]

    title_col, delete_col = st.sidebar.columns([5, 1])

    with title_col:
        if st.button(
            title,
            key=f"open_{thread_id}",
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            switch_to_thread(thread_id)

    with delete_col:
        if st.button(
            "🗑️",
            key=f"delete_{thread_id}",
            use_container_width=True,
            help="Delete this conversation"
        ):
            remove_thread(thread_id)

            # If the deleted thread was the active one, start a fresh chat
            if is_active:
                reset_chat()

            st.rerun()


# ========================= Main chat interface =========================

# Display all messages from the currently selected conversation
for message in st.session_state["message_history"]:

    # Create either a user chat bubble or assistant chat bubble
    with st.chat_message(message["role"]):

        # Display the message content as rendered Markdown (bold text,
        # tables, lists, etc. from RAG/tool-derived answers need this to
        # actually render -- st.text() would show raw "**bold**" and
        # "| pipe | table |" syntax literally instead of formatting it).
        st.markdown(message["content"])

        # Re-display a previously generated image, if this message has one
        image_path = message.get("image_path")
        if image_path and os.path.exists(image_path):
            st.image(image_path)


# ========================= HITL approval interface =========================

# Get the currently pending approval request
pending_hitl = st.session_state.get(
    "pending_hitl"
)

# Check whether the pending approval belongs to
# the currently selected conversation
current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id")
    == st.session_state["thread_id"]
)


# Display approval controls
if current_thread_has_pending_hitl:

    st.warning(
        "🧑 Human approval required\n\n"
        f"{pending_hitl['prompt']}"
    )

    approve_column, reject_column = st.columns(2)

    # Approve button
    with approve_column:

        if st.button(
            "✅ Approve Purchase",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            use_container_width=True
        ):

            # Send "yes" back to interrupt()
            resume_hitl_execution("yes")

    # Reject button
    with reject_column:

        if st.button(
            "❌ Reject Purchase",
            key=f"reject_{st.session_state['thread_id']}",
            use_container_width=True
        ):

            # Send "no" back to interrupt()
            resume_hitl_execution("no")


# ========================= Fixed chat input with PDF upload =========================

# Keep st.chat_input directly in the main body.
# This keeps it fixed at the bottom of the screen.
#
# accept_file=True adds the attachment button inside the chat input.
# file_type=["pdf"] allows PDF files only.
submission = st.chat_input(
    "Type here",
    accept_file=True,
    file_type=["pdf"],

    # Disable input while waiting for human approval
    disabled=current_thread_has_pending_hitl
)


# Default user input value
user_input = None


# Process the submitted text and PDF
if submission:

    # Get the text entered by the user
    user_input = submission.text

    # Get the uploaded files
    # This is always a list when accept_file is enabled
    uploaded_files = submission.files

    # Process the uploaded PDF if one was attached
    if uploaded_files:

        uploaded_pdf = uploaded_files[0]

        # Store the temporary file path
        temporary_file_path = None

        # Read the raw bytes ONCE up front so we can sanity-check them
        # before doing any real work. Picking a file via Google Drive's
        # in-browser picker (as opposed to local storage/Downloads) can
        # hand the browser a truncated or empty file if Drive hasn't
        # finished streaming it down to the device yet -- the picker UI
        # still shows the correct filename/size, but the actual bytes
        # received can be incomplete. Catching that here gives a clear,
        # specific message instead of a confusing low-level PDF parsing
        # error (or being misdiagnosed as "this PDF is scanned").
        raw_pdf_bytes = uploaded_pdf.getvalue()

        # A real PDF is always at least a few hundred bytes, and the
        # "%PDF-" header must appear somewhere in roughly the first KB
        # per the PDF spec (some files have a small amount of leading
        # junk/whitespace before it).
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

                # Save the uploaded PDF as a temporary local file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temporary_file:

                    temporary_file.write(raw_pdf_bytes)

                    temporary_file_path = temporary_file.name

                # Call the existing backend RAG ingestion function
                with st.spinner(
                    f"Processing {uploaded_pdf.name}..."
                ):

                    ingest_rag_document(
                        temporary_file_path,
                        session_id=st.session_state["session_id"],
                        thread_id=st.session_state["thread_id"],
                        filename=uploaded_pdf.name
                    )

                # Display PDF processing confirmation
                st.toast(
                    f"{uploaded_pdf.name} processed successfully.",
                    icon="✅"
                )

            except Exception as error:

                # Display PDF processing error
                st.error(
                    f"PDF processing failed: {error}"
                )

            finally:

                # Delete the temporary PDF after indexing
                if (
                    temporary_file_path
                    and os.path.exists(temporary_file_path)
                ):
                    os.remove(temporary_file_path)


# Run this block after the user submits a text message
if user_input:

    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    # Give this conversation a readable sidebar title right away,
    # instead of waiting for another checkpoint read (ChatGPT/Claude-style)
    cache_thread_title(st.session_state["thread_id"], user_input)

    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.markdown(user_input)

    # Pass the current thread ID (for memory) and session ID (for
    # per-user document scoping in rag_tool) to LangGraph
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

    # Assistant streaming block
    with st.chat_message("assistant"):

        # Use a mutable holder so the generator can set/modify it
        status_holder = {
            "box": None
        }

        # Captures the file path of an image produced by the
        # generate_image tool during this turn, if any, so it can be
        # rendered with st.image() after streaming finishes (the actual
        # image bytes never pass through the LLM's own generated text --
        # see the "IMAGE_FILE::" sentinel handling below).
        generated_image_holder = {
            "path": None
        }

        def ai_only_stream():

            # Tracks whether chat_node actually produced any streamed
            # text. If the guardrail blocks a message, chat_node never
            # runs, so nothing gets streamed here at all -- the fallback
            # below then surfaces the refusal message instead of leaving
            # a blank response.
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

                    # Lazily create & update the SAME status container
                    # when any tool runs
                    if isinstance(
                        message_chunk,
                        ToolMessage
                    ):

                        tool_name = getattr(
                            message_chunk,
                            "name",
                            "tool"
                        )

                        # generate_image returns a sentinel string
                        # ("IMAGE_FILE::<path>") rather than a URL/base64
                        # blob, so the LLM never has to reproduce the raw
                        # image data in its final answer. Capture the path
                        # here so it can be rendered directly with
                        # st.image() once streaming finishes.
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

                    # Stream ONLY assistant tokens that came from
                    # chat_node itself. Without this "langgraph_node"
                    # check, the guardrail's internal security classifier
                    # call (which also invokes the LLM, inside
                    # guardrail_node) would leak its raw "CATEGORY: ...
                    # CONFIDENCE: ..." verdict into the visible response,
                    # since stream_mode="messages" surfaces tokens from
                    # ANY chat model call made anywhere in the graph run.
                    if (
                        isinstance(message_chunk, AIMessage)
                        and metadata.get("langgraph_node") == "chat_node"
                    ):
                        yielded_any_content = True
                        yield message_chunk.content

            except Exception:
                # Last line of defense: chat_node already retries and
                # falls back internally, but if anything else in the
                # graph (a tool call, the streaming layer itself, etc.)
                # still raises, show a normal chat message instead of a
                # raw traceback that crashes the whole app.
                yield (
                    "Sorry, something went wrong while generating a "
                    "response. Please try again, or rephrase your question."
                )
                return

            # The guardrail blocked this message before chat_node ever
            # ran, so nothing was streamed above. Pull the refusal that
            # guardrail_node saved to the checkpoint and show that instead
            # of leaving a blank assistant bubble.
            if not yielded_any_content:

                final_state = chatbot.get_state(config=CONFIG)
                final_messages = final_state.values.get("messages", [])

                if final_messages and isinstance(final_messages[-1], AIMessage):
                    fallback_content = final_messages[-1].content

                    if fallback_content:
                        yield fallback_content

            # ========================= HITL ADDED =========================

            # interrupt() pauses the graph without returning
            # a completed ToolMessage.
            #
            # Inspect the saved checkpoint after streaming ends.
            pending_interrupt = get_pending_interrupt(
                st.session_state["thread_id"]
            )

            if pending_interrupt is not None:

                # Save the interrupt for displaying approval buttons
                save_pending_interrupt(
                    st.session_state["thread_id"],
                    pending_interrupt
                )

                yield (
                    "\n\n⚠️ This stock purchase requires your approval. "
                    "Use the Approve Purchase or Reject Purchase "
                    "button below."
                )

            # =============================================================

        ai_message = st.write_stream(
            ai_only_stream()
        )

        # Display the generated image (if this turn produced one) right
        # after the text answer, in the same assistant bubble.
        if generated_image_holder["path"] and os.path.exists(generated_image_holder["path"]):
            st.image(generated_image_holder["path"])

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:

            # Check whether execution is waiting for approval
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

    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message,
        "image_path": generated_image_holder["path"]
    })

    # ========================= HITL ADDED =========================

    # Approval controls are rendered earlier in the script.
    # Rerun so they appear immediately after interrupt().
    if (
        st.session_state.get("pending_hitl") is not None
        and st.session_state["pending_hitl"].get("thread_id")
        == st.session_state["thread_id"]
    ):
        st.rerun()

    # =============================================================

    # Refresh so the sidebar reflects the new/updated conversation title
    st.rerun()