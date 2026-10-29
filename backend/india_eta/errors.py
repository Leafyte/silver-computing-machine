"""Stable, non-echoing errors for India ETA boundaries."""


class EtaError(ValueError):
    def __init__(self, code: str, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def as_dict(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": self.message}}


def invalid(code: str = "invalid_request", message: str = "Request is invalid") -> EtaError:
    return EtaError(code, message, 422)
