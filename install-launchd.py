#!/usr/bin/env python3
"""Install Breeze as a macOS LaunchAgent with full configuration support."""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict
import plistlib

try:
    from jinja2 import Template
except ImportError:
    print("Error: jinja2 is required. Install it with: pip install jinja2")
    sys.exit(1)


class BreezeInstaller:
    """Installer for Breeze MCP server as a macOS LaunchAgent."""

    def __init__(self):
        self.working_dir = Path.home() / "Library" / "Application Support" / "com.github.casualjim.breeze"
        self.launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
        self.plist_template = Path("com.github.casualjim.breeze.server.plist.template")
        self.plist_file = self.working_dir / "com.github.casualjim.breeze.server.plist"
        self.dest_plist = self.launch_agents_dir / "com.github.casualjim.breeze.server.plist"

        # Config file path
        self.config_file = Path.home() / ".config" / "breeze" / "config.toml"

        # Log directory
        self.log_dir = os.environ.get("LOG_DIR", Path.home() / "Library" / "Logs")

    def check_requirements(self) -> bool:
        """Check if all requirements are met."""
        # Check if template exists
        if not self.plist_template.exists():
            print(f"Error: Template file {self.plist_template} not found")
            return False

        # Check if config file exists
        if not self.config_file.exists():
            print(f"Error: Config file {self.config_file} not found")
            print("Please create a config file in ~/.config/breeze/config.toml")
            return False

        return True

    def create_directories(self):
        """Create necessary directories."""
        self.working_dir.mkdir(parents=True, exist_ok=True)
        # Create LaunchAgents directory
        self.launch_agents_dir.mkdir(parents=True, exist_ok=True)

        # Create log directory
        log_dir = Path(self.log_dir)
        # Try to create without sudo first
        log_dir.mkdir(parents=True, exist_ok=True)

    def generate_plist(self) -> Dict:
        """Generate the plist configuration using Jinja2."""
        # Read template
        with open(self.plist_template, "r") as f:
            template = Template(f.read())

        # Create context with template variables
        context = {
            "CONFIG_FILE": str(self.config_file),
            "WORKING_DIR": str(self.working_dir),
            "LOG_DIR": self.log_dir,
            "HOME": Path.home(),
        }

        # Render template
        plist_content = template.render(context)

        # Write generated plist
        with open(self.plist_file, "w") as f:
            f.write(plist_content)

        # Parse and return as dict for display
        with open(self.plist_file, "rb") as f:
            return plistlib.load(f)

    def install_service(self):
        """Install the LaunchAgent."""
        # Unload existing agent if present
        if (
            subprocess.run(
                ["launchctl", "list"], capture_output=True, text=True
            ).stdout.find("com.breeze-mcp.server")
            != -1
        ):
            print("Unloading existing agent...")
            subprocess.run(
                ["launchctl", "unload", str(self.dest_plist)], stderr=subprocess.DEVNULL
            )

        # Copy plist file
        shutil.copy2(self.plist_file, self.dest_plist)

        # Load the agent
        subprocess.run(["launchctl", "load", str(self.dest_plist)], check=True)

    def print_configuration(self):
        """Print the current configuration."""
        print("\nBreeze MCP server installed and started as a LaunchAgent")
        print("\nConfiguration:")
        print(f"  Config file: {self.config_file}")
        print(f"  Working directory: {self.working_dir}")
        print(f"  Logs: {self.log_dir}/breeze-mcp.log")

        print("\nCommands:")
        print("  Check status: launchctl list | grep breeze")
        print(f"  Stop: launchctl unload {self.dest_plist}")
        print(f"  Start: launchctl load {self.dest_plist}")
        print(f"  View logs: tail -f {self.log_dir}/breeze-mcp.log")

    def run(self):
        """Run the installation process."""
        print("Breeze MCP Server Installer")
        print("=" * 40)

        # Check requirements
        if not self.check_requirements():
            sys.exit(1)

        # Create directories
        print("\nCreating directories...")
        self.create_directories()

        # Generate plist
        print("Generating plist file from template...")
        self.generate_plist()

        # Install service
        print("Installing LaunchAgent...")
        self.install_service()

        # Print configuration
        self.print_configuration()

        # Clean up generated plist file (optional)
        # self.plist_file.unlink()


def main():
    """Main entry point."""
    installer = BreezeInstaller()
    try:
        installer.run()
    except KeyboardInterrupt:
        print("\nInstallation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
