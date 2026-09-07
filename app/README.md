# App Module

## Overview

The `app/` directory contains the Flask application entry point and related configuration for the MorphGuard API server.

## Components

### app.py (project root)

The main Flask application that:

- Initializes the Flask instance and configures CORS
- Registers API blueprints (pipeline_bp, sources_bp)
- Implements authentication endpoints (register, login)
- Sets up MongoDB indexes
- Provides global error handling (JSON responses)
- Exposes the health check endpoint

### Configuration

The application reads configuration from environment variables:

| Variable       | Default                  | Description                    |
|---------------|--------------------------|--------------------------------|
| JWT_SECRET    | change-this-secret       | Secret for signing JWT tokens  |
| CORS_ORIGIN   | http://localhost:3000    | Allowed CORS origin            |
| FLASK_DEBUG   | (off)                    | Enable Flask debug mode        |

### Running

```bash
# Development
python3 app.py

# Development with debug mode
FLASK_DEBUG=1 python3 app.py

# Production (with Gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### API Endpoints Handled Directly

| Endpoint            | Method | Description             |
|---------------------|--------|-------------------------|
| /api/auth/register  | POST   | User registration       |
| /api/auth/login     | POST   | User authentication     |
| /api/health         | GET    | Health check            |

All other endpoints are delegated to the registered blueprints:
- `pipeline_bp` → `/api/pipeline/*`
- `sources_bp` → `/api/sources/*`
