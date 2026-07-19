# Security Policy

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository. Do not open a public issue containing exploit details, credentials, private prompts, or exported model traces. Include the affected commit, deployment mode, reproduction steps, and potential impact.

## Scope and limitations

J-Steer is an interpretability and intervention demo. It is not a production content-safety system, data-loss-prevention boundary, or substitute for output filtering, authorization, audit controls, and human review.

The public Modal deployment has no user accounts. GPU requests are serialized per container and deployment scale is capped, but a public fork should add organization-appropriate authentication, quotas, monitoring, and abuse controls before processing untrusted traffic.

Prompts, generated text, hidden-state readouts, and JSON exports may contain sensitive data. Use synthetic examples only. Do not enter API keys, credentials, personal data, customer records, or proprietary documents.

## Supported versions

Security fixes are applied to the latest commit on `main`. This project does not currently maintain release branches.
