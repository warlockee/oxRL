"""
Ray utility wrappers for fault tolerance.
"""


def ray_get_with_timeout(futures, timeout_sec=0, description="ray.get"):
    """Wrapper around ray.get() that adds timeout support.

    Args:
        futures: List of Ray ObjectRefs (or single ObjectRef).
        timeout_sec: Timeout in seconds. 0 means no timeout (current behavior).
        description: Human-readable label for error messages.

    Returns:
        Results from ray.get().

    Raises:
        TimeoutError: If timeout_sec > 0 and the call exceeds the timeout.
    """
    import ray

    if timeout_sec > 0:
        try:
            return ray.get(futures, timeout=timeout_sec)
        except ray.exceptions.GetTimeoutError:
            n = len(futures) if isinstance(futures, list) else 1
            raise TimeoutError(
                f"{description} timed out after {timeout_sec}s. "
                f"This usually indicates a worker crash or GPU hang. "
                f"Pending futures: {n}"
            )
    return ray.get(futures)
