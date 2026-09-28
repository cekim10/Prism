"""Shim: FastTTS's worker imports pycuda.driver only to print the current CUDA context.
Real pycuda needs the CUDA toolkit headers to build; this stub avoids that dependency."""


class Context:
    @staticmethod
    def get_current():
        return None
