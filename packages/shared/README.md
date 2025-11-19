# @writeo/shared

Shared TypeScript types, schemas, and utilities for the Writeo project.

## TypeScript Package

### Installation

The package is automatically available to other packages in the monorepo via npm workspaces.

### Building

```bash
cd packages/shared
npm run build
```

This compiles TypeScript files from `ts/` to `dist/`.

### Usage

```typescript
import {
  CreateQuestionRequest,
  CreateAnswerRequest,
  CreateSubmissionRequest,
  ModalRequest,
  AssessmentResults,
  isValidUUID,
  mapScoreToCEFR,
} from "@writeo/shared";
```

### Exports

- **Types**: Request/response types for API endpoints
- **Validation**: UUID validation utilities
- **CEFR Mapping**: Band score to CEFR level conversion

## Python Package

### Installation

The Python package can be installed as an editable package:

```bash
cd packages/shared/py
pip install -e .
```

Or in Modal, it's automatically installed in the image build process.

### Usage

```python
from schemas import (
    ModalRequest,
    AssessmentResults,
    AssessmentPart,
    AssessorResult,
    map_score_to_cefr,
)
```

### Structure

- `schemas.py` - Pydantic models for Modal service
- `__init__.py` - Package exports
- `pyproject.toml` - Package configuration

## Development

### TypeScript

1. Make changes in `ts/` directory
2. Run `npm run build` to compile
3. Changes are automatically available to consuming packages

### Python

1. Make changes in `py/` directory
2. For Modal: changes are picked up automatically (editable install)
3. For local development: run `pip install -e .` in `py/` directory

## Notes

- TypeScript package must be built before other packages can use it
- Python package uses Pydantic v2 for schema validation
- Both packages maintain the same data structures for consistency
