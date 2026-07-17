export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const ALLOWED_FILE_TYPES = ['.txt', '.md', '.csv', '.json'];
export const ALLOWED_MIME_TYPES = ['text/plain', 'text/markdown', 'text/csv', 'application/json'];

export const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB per file
