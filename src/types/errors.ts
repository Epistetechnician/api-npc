export class ValidationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ValidationError'
  }
}

export class DataIngestionError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'DataIngestionError'
  }
}

export class APIError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'APIError'
  }
} 