"""
Communication simulation package for ARCE in OpenCOOD / OPV2V.

This package contains the communication-side modules used by
point_pillar_transformer_opv2v_arce.py, including:

1. channel
   - fixed Good / Medium / Bad channel profiles
   - Gilbert-Elliott packet loss model
   - latency estimation
   - channel manager

2. packet
   - feature packetization
   - packet metadata
   - communication size estimation

3. fec
   - no-FEC baseline
   - XOR parity recovery
   - Raptor / fountain-code simulation

4. recovery
   - zero-fill recovery
   - spatial interpolation
   - temporal feature cache
   - partial reconstruction controller

5. arce
   - fixed ARCE policy
   - ARCE communication pipeline

6. metrics
   - communication statistics
   - per-frame / per-link logging

Design note:
    Keep this __init__.py lightweight.

    Do NOT import heavy submodules here, such as ARCEFixedComm,
    GilbertElliott, FeaturePacketizer, or FEC modules. Those modules
    may depend on torch, numpy, or other OpenCOOD components, and eager
    imports here can easily introduce circular imports or import errors
    while the project is still under development.

    Concrete modules should be imported directly where they are used.
    For example:

        from opencood.comm.arce.arce_fixed_comm import ARCEFixedComm

    instead of importing ARCEFixedComm from opencood.comm.
"""

PACKAGE_NAME = "opencood.comm"
PACKAGE_VERSION = "0.1.0"


def get_package_info():
    """
    Return basic information about the communication package.

    This is mainly useful for debugging, logging, and checking whether
    the ARCE communication package has been correctly installed or added
    to the project tree.

    Returns
    -------
    dict
        Basic package metadata.
    """
    return {
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "description": (
            "Communication simulation package for ARCE, including "
            "channel modeling, packetization, quantization-aware "
            "transmission, FEC, partial reconstruction, temporal cache, "
            "and communication logging."
        ),
    }


__all__ = [
    "PACKAGE_NAME",
    "PACKAGE_VERSION",
    "get_package_info",
]