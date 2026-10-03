from typing import Literal
from pydantic import BaseModel, Field

class category(BaseModel) :
    category : Literal['technical', 'news', 'casual'] = Field(description='Category of the given topic')