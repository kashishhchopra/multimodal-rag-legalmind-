"""Typed application errors so the API layer can map them to clean HTTP codes.

Every raised error carries a stable `code` string the frontend can switch on,
plus a human message shown in the UI (spec section 16).
"""
from __future__ import annotations


class AppError(Exception):
    code: str = "app_error"
    http_status: int = 400

    def __init__(self, message: str, *, code: str | None = None, http_status: int | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if http_status:
            self.http_status = http_status

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


class InvalidFileError(AppError):
    code = "invalid_file"
    http_status = 400


class UnsupportedFileTypeError(AppError):
    code = "unsupported_file_type"
    http_status = 415


class EmptyDocumentError(AppError):
    code = "empty_document"
    http_status = 422


class DocumentTooLargeError(AppError):
    code = "document_too_large"
    http_status = 413


class DocumentNotFoundError(AppError):
    code = "document_not_found"
    http_status = 404


class DocumentNotReadyError(AppError):
    code = "document_not_ready"
    http_status = 409


class SourceNotFoundError(AppError):
    code = "source_not_found"
    http_status = 404


class OCRUnavailableError(AppError):
    code = "ocr_unavailable"
    http_status = 500


class EmbeddingError(AppError):
    code = "embedding_failed"
    http_status = 500


class VectorStoreError(AppError):
    code = "vector_store_failed"
    http_status = 500


class LLMError(AppError):
    code = "llm_failed"
    http_status = 502


class MissingAPIKeyError(AppError):
    code = "missing_api_key"
    http_status = 400
