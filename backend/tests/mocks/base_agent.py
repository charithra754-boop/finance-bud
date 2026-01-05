"""
Base Agent for Mocks
Imports the real BaseAgent to ensure compatibility.
"""
import sys
import os

# Ensure backend directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from agents.base_agent import BaseAgent
