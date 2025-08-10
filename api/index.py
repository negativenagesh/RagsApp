import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "mcp_server"))
from main import mcp

app = mcp.app