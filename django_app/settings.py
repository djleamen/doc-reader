"""
Django settings for django_app project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'django-insecure-change-me-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'rag_app',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'django_app.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'django_app.wsgi.application'

# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20
}

# CORS settings
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

CORS_ALLOW_ALL_ORIGINS = DEBUG

# RAG Engine settings (from existing .env)
RAG_SETTINGS = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
    'PINECONE_ENVIRONMENT': os.getenv('PINECONE_ENVIRONMENT'),
    'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002'),
    'CHAT_MODEL': os.getenv('CHAT_MODEL', 'gpt-4-turbo-preview'),
    'MAX_TOKENS': int(os.getenv('MAX_TOKENS', '4000')),
    'TEMPERATURE': float(os.getenv('TEMPERATURE', '0.1')),
    'VECTOR_DB_TYPE': os.getenv('VECTOR_DB_TYPE', 'faiss'),
    'CHUNK_SIZE': int(os.getenv('CHUNK_SIZE', '1000')),
    'CHUNK_OVERLAP': int(os.getenv('CHUNK_OVERLAP', '200')),
    'TOP_K_RESULTS': int(os.getenv('TOP_K_RESULTS', '5')),
    'MAX_DOCUMENT_SIZE_MB': int(os.getenv('MAX_DOCUMENT_SIZE_MB', '100')),
    'SUPPORTED_FORMATS': os.getenv('SUPPORTED_FORMATS', 'pdf,docx,txt,md').split(','),
}

# File upload settings
FILE_UPLOAD_MAX_MEMORY_SIZE = RAG_SETTINGS['MAX_DOCUMENT_SIZE_MB'] * 1024 * 1024  # MB to bytes
DATA_UPLOAD_MAX_MEMORY_SIZE = FILE_UPLOAD_MAX_MEMORY_SIZE

# Directory settings
DOCUMENTS_DIR = BASE_DIR / 'documents'
INDEXES_DIR = BASE_DIR / 'indexes'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(INDEXES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
