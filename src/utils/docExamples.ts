export const DOCUMENTATION_SECTIONS = [
  {
    id: 'overview',
    title: 'Overview',
    icon: Book,
  },
  // Add other sections...
]

export const API_REFERENCE = {
  endpoints: [
    {
      name: 'Example Endpoint',
      description: 'Description of the endpoint',
      parameters: [
        {
          name: 'param1',
          type: 'string',
          description: 'Description of param1',
          required: true,
        },
      ],
      examples: [
        {
          title: 'Example 1',
          code: 'example code',
          response: 'example response',
        },
      ],
      rateLimit: '100 requests per minute',
    },
  ],
} 