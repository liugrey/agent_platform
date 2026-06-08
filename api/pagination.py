"""
api/pagination.py
Reusable pagination query and response models for API endpoints.
"""

from __future__ import annotations

from math import ceil
from typing import Generic, TypeVar

from pydantic import Field

from api.camel_model import CamelModel

T = TypeVar("T")


class PaginationQuery(CamelModel):
    page: int = Field(default=1, ge=1, description="页码，从 1 开始")
    page_size: int = Field(default=20, ge=1, le=100, description="每页数量")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class PaginationMeta(CamelModel):
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页数量")
    total: int = Field(..., description="总记录数")
    total_pages: int = Field(..., description="总页数")


class PaginatedResponse(CamelModel, Generic[T]):
    items: list[T] = Field(default_factory=list, description="当前页数据")
    pagination: PaginationMeta

    @classmethod
    def create(cls, *, items: list[T], page: int, page_size: int, total: int) -> "PaginatedResponse[T]":
        total_pages = ceil(total / page_size) if total > 0 else 0
        return cls(
            items=items,
            pagination=PaginationMeta(
                page=page,
                page_size=page_size,
                total=total,
                total_pages=total_pages,
            ),
        )
