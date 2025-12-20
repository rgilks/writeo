# Documentation Index

This directory contains all documentation for the Writeo project.

---

## 📚 Quick Start

**New to Writeo?** Start here:

1. **[../README.md](../README.md)** - Project overview and quick start guide
2. **[Operations Guide](operations/deployment.md)** - Deploy your own instance
3. **[Status](reference/status.md)** - Current status and roadmap

---

## 📖 Documentation Structure

### [Architecture](architecture/overview.md)

Detailed system design and API documentation.

- **[Overview](architecture/overview.md)** - System architecture, components, data flow
- **[API Worker](architecture/api-worker.md)** - Detailed API Worker architecture
- **[State Management](architecture/state-management.md)** - Frontend state management

### [Guides](guides/services.md)

How-to guides and service documentation.

- **[Services Overview](guides/services.md)** - Documentation for Modal services
- **[Adding Services](guides/adding-services.md)** - Guide for adding new Modal assessor services

### [Models](models/overview.md)

ML models and AI/LLM integration details.

- **[Overview](models/overview.md)** - Essay scoring models comparison
- **[Corpus Model](models/corpus.md)** - AES-CORPUS training details
- **[DeBERTa Model](models/deberta.md)** - AES-DEBERTA (DeBERTa v3 Large) details
- **[Feedback Model](models/feedback.md)** - AES-FEEDBACK multi-task model
- **[GEC Models](models/gec.md)** - Grammar correction services (Seq2Seq + GECToR)
- **[Datasets](models/datasets.md)** - Available datasets for training
- **[Evaluations](models/evaluation.md)** - Performance evaluation report

### [Operations](operations/monitoring.md)

Deployment, monitoring, and maintenance.

- **[Deployment](operations/deployment.md)** - Step-by-step deployment guide
- **[Monitoring & Ops](operations/monitoring.md)** - Logging, monitoring, performance tuning
- **[Cost Analysis](operations/cost.md)** - Cost analysis and pricing guardrails
- **[Operational Modes](operations/modes.md)** - Switching between Cheap and Turbo modes
- **[Scripts](operations/scripts.md)** - Scripts reference
- **[Training Guide](operations/training.md)** - Model training pipeline
- **[PWA Setup](operations/pwa.md)** - PWA configuration
- **[Testing](operations/testing.md)** - Testing strategies

### [Reference](reference/openapi.yaml)

Specifications, legal, and meta-docs.

- **[OpenAPI Spec](reference/openapi.yaml)** - OpenAPI 3.0 specification
- **[Legal](reference/legal.md)** - Compliance and legal checks
- **[Status](reference/status.md)** - Project status and roadmap

---

## 🔗 Key Links

| Topic          | Link                                                                           |
| -------------- | ------------------------------------------------------------------------------ |
| **API Docs**   | [Interactive Swagger UI](https://writeo-api-worker.rob-gilks.workers.dev/docs) |
| **API Spec**   | [openapi.yaml](reference/openapi.yaml)                                         |
| **Deployment** | [operations/deployment.md](operations/deployment.md)                           |
| **Cost**       | [operations/cost.md](operations/cost.md)                                       |
| **Models**     | [models/overview.md](models/overview.md)                                       |

---

## 📝 Standards

- Keep documentation current with code changes.
- Consolidate new findings into existing sections (e.g. `models/overview.md`).
- Use relative links for navigation.
