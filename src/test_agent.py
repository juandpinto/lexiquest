#!/usr/bin/env python3
"""
Wrapper script to run any agent module as a script, ensuring proper package context.
Usage:
    python run_agent.py assessment_agent
    python run_agent.py manager_agent
    python run_agent.py alignment_agent
"""

import sys
import subprocess
import os

# Add the src directory to sys.path so 'agents' can be found
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

if len(sys.argv) != 2:
    print("Usage: python run_agent.py <agent_module>")
    print("Example: python run_agent.py assessment_agent")
    sys.exit(1)

agent_module = sys.argv[1]
module_path = f"agents.{agent_module}"


try:
    # Set working directory to src so Python can find the agents package
    src_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, "-m", module_path], check=True, cwd=src_dir)
except subprocess.CalledProcessError as e:
    print(f"Error running agent module '{module_path}': {e}")
    sys.exit(e.returncode)
