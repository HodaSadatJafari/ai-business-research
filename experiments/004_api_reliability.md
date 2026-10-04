# Experiment 004 — Research API reliability

**Date:** September 29, 2026
**Status:** First Week 4 production-engineering slice implemented and tested locally.

## Hypothesis

Research endpoints should return stable, safe HTTP responses when the model
provider fails. The baseline endpoint should also return a successful response
when its synchronous service succeeds.

## Change

The baseline route now runs as a synchronous FastAPI route, matching its
synchronous service. All three research routes use one error boundary:
connection failures and rate limits return HTTP 503; other OpenAI SDK errors
return HTTP 502. Error bodies use fixed messages so upstream details are not
sent to clients. The existing tool-unsupported response remains HTTP 502.

## Measurement

Using FastAPI TestClient and stubbed services, with no live model or Tavily
calls:

| Scenario | Before | After |
| --- | ---: | ---: |
| Successful baseline service response | HTTP 500 | HTTP 200 |
| Model connection failure, baseline | HTTP 500 | HTTP 503 |
| Model connection failure, RAG | HTTP 500 | HTTP 503 |
| Model connection failure, agent | HTTP 500 | HTTP 503 |

The before values were observed against commit `a817e9b`. Deterministic tests
also verify that rate limits return 503, other provider status errors return
502, and none of the three endpoints returns a simulated private upstream
error message. The full test suite grew from 22 to 32 passing tests. Ruff
passes.

## Limits

These are local failure simulations. They establish API behavior, not recovery
during a real provider outage. This slice does not change the OpenAI SDK's
retry or timeout settings, measure latency, or evaluate research quality.
