from typing import List, Literal, Optional

from pydantic import BaseModel


class Plan(BaseModel):
    intent: Literal["compare", "reviews", "alternatives", "recommend",
                    "clarify", "off_topic"]
    product_mentions: List[str]
    aspects: List[str]
    max_price: Optional[float]
    brand: Optional[str]
    use_case: Optional[str]
    clarification_question: Optional[str]
