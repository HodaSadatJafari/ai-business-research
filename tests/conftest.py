"""Keep unit tests independent of the configured Opik service."""

import app.observability

app.observability.configure_opik = lambda: None
