# Core framework
fastapi==0.115.0
uvicorn[standard]==0.30.6
websockets==13.1
python-multipart==0.0.12

# Anthropic
anthropic==0.40.0

# STT / TTS
deepgram-sdk==3.7.5
elevenlabs==1.9.0

# Telephony
twilio==9.3.5

# Database
sqlalchemy[asyncio]==2.0.36
asyncpg==0.30.0
alembic==1.13.3
psycopg2-binary==2.9.10

# Redis
redis[hiredis]==5.2.0

# Task queue
celery==5.4.0
celery[redis]==5.4.0

# Language detection
fasttext-wheel==0.9.2

# Validation & config
pydantic==2.10.3
pydantic-settings==2.6.1
python-dotenv==1.0.1

# Audio processing
numpy==2.2.0
scipy==1.14.1

# HTTP client
httpx==0.28.1
aiohttp==3.11.10

# Logging & observability
structlog==24.4.0
prometheus-client==0.21.1

# Utilities
python-jose[cryptography]==3.3.0
pendulum==3.0.0
tenacity==9.0.0

# Testing
pytest==8.3.4
pytest-asyncio==0.24.0
pytest-mock==3.14.0
httpx==0.28.1
fakeredis[aioredis]==2.26.1
