def update_memory(memory, plan):
    """Remember budget, brand and use case across messages."""
    if plan.max_price:
        memory["budget"] = plan.max_price
    if plan.brand:
        memory["brand"] = plan.brand
    if plan.use_case:
        memory["use_case"] = plan.use_case
    return memory
