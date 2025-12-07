import os
from typing import Any

import modal

# Default configuration constants
DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes
DEFAULT_MEMORY_MB = 1024
DEFAULT_VOLUME_NAME = "writeo-volume"
DEFAULT_VOLUME_MOUNT = "/vol"
DEFAULT_SCALEDOWN_WINDOW_SECONDS = 10
DEFAULT_GPU_TYPE = "T4"


class ModalServiceFactory:
    """
    Factory for creating consistent Modal service configurations.
    Reduces boilerplate for app creation, image definition, and volume mounting.
    """

    @staticmethod
    def create_app(
        name: str,
        image_python_version: str = "3.12",
        system_packages: list[str] | None = None,
        pip_packages: list[str] | None = None,
        include_shared_package: bool = True,
        app_dir: str | None = None,
    ) -> tuple[modal.App, modal.Image]:
        """
        Create a standardized Modal App and Image.
        """
        app = modal.App(name)

        image = modal.Image.debian_slim(python_version=image_python_version)

        if system_packages:
            image = image.apt_install(system_packages)

        if pip_packages:
            image = image.pip_install(pip_packages)

        # Mount local shared package if requested
        if include_shared_package:
            # Resolve path to packages/shared/py relative to this file
            try:
                # This file is in packages/shared/py
                shared_pkg_path = os.path.dirname(os.path.abspath(__file__))

                # Check if we are in the right place (validating by presence of this file)
                if os.path.exists(os.path.join(shared_pkg_path, "modal_utils.py")):
                    # Mount the directory explicitly to avoid module resolution issues
                    # and ensure it is in the PYTHONPATH
                    remote_path = "/root/shared-pkg"
                    image = image.add_local_dir(shared_pkg_path, remote_path=remote_path)
                    image = image.env({"PYTHONPATH": remote_path})
            except Exception:
                # Fallback or ignore if path resolution fails
                pass

        # Mount local app directory if provided
        if app_dir:
            image = image.add_local_dir(app_dir, remote_path="/app", copy=True)

        # Set environment variables if needed
        # image = image.env({"ENV_VAR": "VALUE"})

        return app, image

    @staticmethod
    def get_default_function_kwargs(
        image: modal.Image,
        volume_mount: str = DEFAULT_VOLUME_MOUNT,
        volume_name: str = DEFAULT_VOLUME_NAME,
        gpu: str | Any = DEFAULT_GPU_TYPE,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        memory: int = DEFAULT_MEMORY_MB,
        scaledown_window: int = DEFAULT_SCALEDOWN_WINDOW_SECONDS,
        secrets: list[modal.Secret] | None = None,
        cpu: float | None = None,
        volumes: dict[str, modal.Volume | modal.NetworkFileSystem] | None = None,
    ) -> dict[str, Any]:
        """
        Get standard kwargs for @app.function decorator.
        """
        # Create default volume if no custom volumes provided
        if volumes is None:
            volume = modal.Volume.from_name(volume_name, create_if_missing=True)
            volumes = {volume_mount: volume}

        kwargs = {
            "image": image,
            "gpu": gpu,
            "timeout": timeout,
            "volumes": volumes,
            "memory": memory,
            "container_idle_timeout": scaledown_window,
        }

        if secrets:
            kwargs["secrets"] = secrets

        if cpu:
            kwargs["cpu"] = cpu

        return kwargs
