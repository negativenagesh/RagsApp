from pydantic import BaseModel
from typing import Dict, Any, List

class Permissions(BaseModel):
    key : str
    value: str
    allowed: bool

class Message(BaseModel):
    params: Dict[str, Any] = {}
    config: Dict[str, Any] = {}
    permissions: List[Permissions] = []