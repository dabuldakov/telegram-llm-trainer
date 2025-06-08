from pydantic import BaseModel

class Request(BaseModel):
    user: str
    prompt: str
    max_length: int = 100