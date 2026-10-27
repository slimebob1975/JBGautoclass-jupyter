"""JBG semantic action states, shared contract for future web frontends.

Ready: green; waiting: yellow; blocked: red and disabled;
warning: orange and actionable. ipywidgets exposes warning via CSS class.
"""

STATES = ("ready", "waiting", "blocked", "warning")
COLORS = {"ready": "#198754", "waiting": "#EAB308", "blocked": "#DC2626", "warning": "#EA580C"}


def set_action_state(button, state):
    if state not in STATES:
        raise ValueError("Invalid action state")
    button.disabled = state in ("blocked", "waiting")
    button.button_style = {"ready": "success", "waiting": "warning", "blocked": "danger", "warning": "warning"}[state]
    if hasattr(button, "style") and hasattr(button.style, "button_color"):
        button.style.button_color = COLORS[state]
    if hasattr(button, "remove_class") and hasattr(button, "add_class"):
        for old in STATES:
            button.remove_class("jbg-action-" + old)
        button.add_class("jbg-action-" + state)
    return button
