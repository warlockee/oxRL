"""
Ray cluster initialization.

Single entry point: setup_ray().
"""
import os
import time
import ray


def setup_ray(ray_address):
    """Initialize Ray cluster and return (ray module, master_addr)."""
    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        import tempfile
        import getpass
        user = getpass.getuser()
        ray_temp_dir = os.path.join("/tmp", f"ray_{user}_{int(time.time())}")
        os.makedirs(ray_temp_dir, exist_ok=True)
        ray.init(ignore_reinit_error=True, _temp_dir=ray_temp_dir)

    try:
        master_addr = ray.util.get_node_ip_address()
    except Exception:
        print("Warning: Could not get master address, using localhost")
        master_addr = "127.0.0.1"

    return ray, master_addr
