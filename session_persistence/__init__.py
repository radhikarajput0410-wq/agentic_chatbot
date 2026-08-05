"""
A minimal, hand-rolled Streamlit custom component that stores a
session_id in the browser's own localStorage.

Why this exists: st.session_state alone does NOT survive a hard page
refresh (a refresh opens a brand-new connection, which Streamlit treats
as a brand-new session). Putting session_id in the URL (st.query_params)
does survive a refresh, but is a security bug -- it makes the session_id
visible and shareable via the address bar, so copying/sending the URL to
someone else hands them your exact session.

localStorage is the correct middle ground: it lives in the visitor's own
browser profile (scoped to this site's origin), survives refreshes, but
is never part of a URL and is never transmitted anywhere -- there is no
way to "share" it to another browser/device by sending a link.

No npm/React build is required: the frontend is a single, invisible
static HTML file that speaks Streamlit's component postMessage protocol
directly.
"""

import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

_session_persistence_component = components.declare_component(
    "session_persistence",
    path=_COMPONENT_DIR,
)


def get_persistent_session_id(key=None):
    """
    Returns a session_id that's stored in the browser's localStorage.

    On the very first script run for a given browser connection, the
    component's JS hasn't executed yet, so this returns None -- callers
    should treat None as "not ready yet" and st.stop() until the
    following automatic rerun (triggered by the component reporting its
    value) provides the real session_id.
    """
    return _session_persistence_component(key=key, default=None)