from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel
from pydantic import BaseConfig
from pydantic.dataclasses import dataclass as pydantic_dataclass
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal #type: ignore

BaseConfig.arbitrary_types_allowed = True


# https://fastapi.tiangolo.com/tutorial/sql-databases/?h=initial#create-initial-pydantic-models-schemas
class SearchRequest(BaseModel):
    image_bytes : str
    top_k: Optional[int]

class SearchResponse(BaseModel):
    result: List[str]