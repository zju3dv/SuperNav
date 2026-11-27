"""Navigation Tool protocols, registry and concrete implementations.

Tools are grouped into navigation, perception, mapping, status
and session modules. Import this package before querying ToolRegistry
to register the complete callable surface.

See docs/release/en/tool-authoring-guide.md for tool authoring.
"""

from supernav.methods.navigation.tools import base  # noqa: F401

# Importing each Tool category module triggers `ToolRegistry.register`
# at module-import time, which is why the explicit imports below matter.
# Do not convert these into `import *` or a star re-export — the order
# and the side effect are the point.
from supernav.methods.navigation.tools import navigation  # noqa: F401
from supernav.methods.navigation.tools import navigation_oracle  # noqa: F401
from supernav.methods.navigation.tools import navigation_localnav  # noqa: F401  — learned point-goal local nav (mapless-capable)
from supernav.methods.navigation.tools import perception  # noqa: F401
from supernav.methods.navigation.tools import mapping  # noqa: F401
from supernav.methods.navigation.tools import status  # noqa: F401
from supernav.methods.navigation.tools import session  # noqa: F401
