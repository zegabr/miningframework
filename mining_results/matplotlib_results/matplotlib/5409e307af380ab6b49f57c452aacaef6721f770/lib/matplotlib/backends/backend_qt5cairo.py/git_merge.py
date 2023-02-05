<<<<<<< ./matplotlib/5409e307af380ab6b49f57c452aacaef6721f770/lib/matplotlib/backends/backend_qt5cairo.py/left.py
from .. import backends

backends._QT_FORCE_QT5_BINDING = True
from .backend_qtcairo import (  # noqa: F401, E402 # pylint: disable=W0611
    _BackendQTCairo, FigureCanvasQTCairo, FigureCanvasCairo, FigureCanvasQT
||||||| ./matplotlib/5409e307af380ab6b49f57c452aacaef6721f770/lib/matplotlib/backends/backend_qt5cairo.py/base.py
from .backend_qtcairo import (
    _BackendQTCairo, FigureCanvasQTCairo,
    FigureCanvasCairo, FigureCanvasQT,
    RendererCairo
=======
from .. import backends

backends._QT_FORCE_QT5_BINDING = True
from .backend_qtcairo import (  # noqa: F401, E402 # pylint: disable=W0611
    _BackendQTCairo, FigureCanvasQTCairo, FigureCanvasCairo, FigureCanvasQT,
    RendererCairo
>>>>>>> ./matplotlib/5409e307af380ab6b49f57c452aacaef6721f770/lib/matplotlib/backends/backend_qt5cairo.py/right.py
)


@_BackendQTCairo.export
class _BackendQT5Cairo(_BackendQTCairo):
    pass
