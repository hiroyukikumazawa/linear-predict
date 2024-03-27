from pydantic import BaseModel


class ValueList(BaseModel):
    values: list[float]
