from bson import ObjectId

class Messages:
    def __init__(self, message, ignore=False):
        self.message = message
        self.ignore = ignore

    def __str__(self):
        return f"Message(message={self.message}, ignore='{self.ignore}')"

    def to_dict(self):
        return {"message": self.message, "ignore": self.ignore}

    def to_dict(self):
        return {"message": self._convert_objectid(self.message), "ignore": self.ignore}

    def _convert_objectid(self, obj):
        """Recursively convert ObjectId to string"""
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, list):
            return [self._convert_objectid(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_objectid(v) for k, v in obj.items()}
        return obj


class FunctionResponse:
    """
    Message:
    Card:
    Failed
    """

    def __init__(
        self, message: Messages, card=None, failed: bool = False, memory={}, jump=-1
    ):
        self.message = message
        self.card = card
        self.failed = failed
        self.memory = memory
        self.jump = jump

    def __str__(self):
        return f"FunctionResponse(message={self.message}, failed='{self.failed}')"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            "message": self.message.to_dict(),  # Convert Message object to dict
            "card": self.card,
            "failed": self.failed,
            "memory": self.memory,
            "jump": self.jump,
        }
