class APIError(Exception):
    def __init__(self, status, code, message, field=None):
        super().__init__(message)
        self.status, self.code, self.field = status, code, field

    def body(self):
        error = {"code": self.code, "message": str(self)}
        if self.field is not None:
            error["field"] = self.field
        return {"error": error}
