"""
Channel modeling package for ARCE communication simulation.

This subpackage contains channel-side modules used by the ARCE
communication layer in OPV2V / V2X-ViT experiments.

Planned modules:

1. fixed_channel.py
   Fixed Good / Medium / Bad channel profiles.

2. gilbert_elliott.py
   Gilbert-Elliott two-state packet loss model.

3. latency_model.py
   Size-bandwidth-based latency estimation with bounded jitter.

4. channel_manager.py
   Unified interface for channel state, packet loss, and latency.

Design note:
    Keep this __init__.py lightweight.

    Do not eagerly import concrete channel classes here, because those
    files may depend on numpy, torch, random generators, or other modules
    that may still be under development.

    Concrete modules should be imported directly where they are used:

        from opencood.comm.channel.channel_manager import ChannelManager
        from opencood.comm.channel.gilbert_elliott import GilbertElliott
        from opencood.comm.channel.fixed_channel import FixedChannel
"""

CHANNEL_STATE_GOOD = "good"
CHANNEL_STATE_MEDIUM = "medium"
CHANNEL_STATE_BAD = "bad"

VALID_CHANNEL_STATES = (
    CHANNEL_STATE_GOOD,
    CHANNEL_STATE_MEDIUM,
    CHANNEL_STATE_BAD,
)

DEFAULT_BANDWIDTH_MBPS = {
    CHANNEL_STATE_GOOD: 27.0,
    CHANNEL_STATE_MEDIUM: 5.0,
    CHANNEL_STATE_BAD: 1.0,
}

DEFAULT_JITTER_MS = {
    CHANNEL_STATE_GOOD: (2.0, 8.0),
    CHANNEL_STATE_MEDIUM: (5.0, 20.0),
    CHANNEL_STATE_BAD: (10.0, 40.0),
}


def normalize_channel_state(state):
    """
    Normalize a channel state string.

    Parameters
    ----------
    state : str
        Input channel state. Expected values are good, medium, or bad.
        Case-insensitive.

    Returns
    -------
    str
        Normalized lower-case channel state.

    Raises
    ------
    ValueError
        If the channel state is not one of good / medium / bad.
    """
    if state is None:
        raise ValueError(
            "Channel state is None. Expected one of: "
            f"{VALID_CHANNEL_STATES}."
        )

    state = str(state).strip().lower()

    if state not in VALID_CHANNEL_STATES:
        raise ValueError(
            f"Invalid channel state: {state}. "
            f"Expected one of: {VALID_CHANNEL_STATES}."
        )

    return state


def is_valid_channel_state(state):
    """
    Check whether a channel state is valid.

    Parameters
    ----------
    state : str
        Input channel state.

    Returns
    -------
    bool
        True if state is one of good / medium / bad, otherwise False.
    """
    try:
        normalize_channel_state(state)
        return True
    except ValueError:
        return False


def get_default_bandwidth_mbps(state):
    """
    Get default bandwidth for a channel state.

    Parameters
    ----------
    state : str
        Channel state: good, medium, or bad.

    Returns
    -------
    float
        Default bandwidth in Mbps.
    """
    state = normalize_channel_state(state)
    return DEFAULT_BANDWIDTH_MBPS[state]


def get_default_jitter_range_ms(state):
    """
    Get default jitter range for a channel state.

    Parameters
    ----------
    state : str
        Channel state: good, medium, or bad.

    Returns
    -------
    tuple
        A tuple of (min_jitter_ms, max_jitter_ms).
    """
    state = normalize_channel_state(state)
    return DEFAULT_JITTER_MS[state]


__all__ = [
    "CHANNEL_STATE_GOOD",
    "CHANNEL_STATE_MEDIUM",
    "CHANNEL_STATE_BAD",
    "VALID_CHANNEL_STATES",
    "DEFAULT_BANDWIDTH_MBPS",
    "DEFAULT_JITTER_MS",
    "normalize_channel_state",
    "is_valid_channel_state",
    "get_default_bandwidth_mbps",
    "get_default_jitter_range_ms",
]"""
Channel modeling package for ARCE communication simulation.

This subpackage contains channel-side modules used by the ARCE
communication layer in OPV2V / V2X-ViT experiments.

Planned modules:

1. fixed_channel.py
   Fixed Good / Medium / Bad channel profiles.

2. gilbert_elliott.py
   Gilbert-Elliott two-state packet loss model.

3. latency_model.py
   Size-bandwidth-based latency estimation with bounded jitter.

4. channel_manager.py
   Unified interface for channel state, packet loss, and latency.

Design note:
    Keep this __init__.py lightweight.

    Do not eagerly import concrete channel classes here, because those
    files may depend on numpy, torch, random generators, or other modules
    that may still be under development.

    Concrete modules should be imported directly where they are used:

        from opencood.comm.channel.channel_manager import ChannelManager
        from opencood.comm.channel.gilbert_elliott import GilbertElliott
        from opencood.comm.channel.fixed_channel import FixedChannel
"""

CHANNEL_STATE_GOOD = "good"
CHANNEL_STATE_MEDIUM = "medium"
CHANNEL_STATE_BAD = "bad"

VALID_CHANNEL_STATES = (
    CHANNEL_STATE_GOOD,
    CHANNEL_STATE_MEDIUM,
    CHANNEL_STATE_BAD,
)

DEFAULT_BANDWIDTH_MBPS = {
    CHANNEL_STATE_GOOD: 27.0,
    CHANNEL_STATE_MEDIUM: 5.0,
    CHANNEL_STATE_BAD: 1.0,
}

DEFAULT_JITTER_MS = {
    CHANNEL_STATE_GOOD: (2.0, 8.0),
    CHANNEL_STATE_MEDIUM: (5.0, 20.0),
    CHANNEL_STATE_BAD: (10.0, 40.0),
}


def normalize_channel_state(state):
    """
    Normalize a channel state string.

    Parameters
    ----------
    state : str
        Input channel state. Expected values are good, medium, or bad.
        Case-insensitive.

    Returns
    -------
    str
        Normalized lower-case channel state.

    Raises
    ------
    ValueError
        If the channel state is not one of good / medium / bad.
    """
    if state is None:
        raise ValueError(
            "Channel state is None. Expected one of: "
            f"{VALID_CHANNEL_STATES}."
        )

    state = str(state).strip().lower()

    if state not in VALID_CHANNEL_STATES:
        raise ValueError(
            f"Invalid channel state: {state}. "
            f"Expected one of: {VALID_CHANNEL_STATES}."
        )

    return state


def is_valid_channel_state(state):
    """
    Check whether a channel state is valid.

    Parameters
    ----------
    state : str
        Input channel state.

    Returns
    -------
    bool
        True if state is one of good / medium / bad, otherwise False.
    """
    try:
        normalize_channel_state(state)
        return True
    except ValueError:
        return False


def get_default_bandwidth_mbps(state):
    """
    Get default bandwidth for a channel state.

    Parameters
    ----------
    state : str
        Channel state: good, medium, or bad.

    Returns
    -------
    float
        Default bandwidth in Mbps.
    """
    state = normalize_channel_state(state)
    return DEFAULT_BANDWIDTH_MBPS[state]


def get_default_jitter_range_ms(state):
    """
    Get default jitter range for a channel state.

    Parameters
    ----------
    state : str
        Channel state: good, medium, or bad.

    Returns
    -------
    tuple
        A tuple of (min_jitter_ms, max_jitter_ms).
    """
    state = normalize_channel_state(state)
    return DEFAULT_JITTER_MS[state]


__all__ = [
    "CHANNEL_STATE_GOOD",
    "CHANNEL_STATE_MEDIUM",
    "CHANNEL_STATE_BAD",
    "VALID_CHANNEL_STATES",
    "DEFAULT_BANDWIDTH_MBPS",
    "DEFAULT_JITTER_MS",
    "normalize_channel_state",
    "is_valid_channel_state",
    "get_default_bandwidth_mbps",
    "get_default_jitter_range_ms",
]