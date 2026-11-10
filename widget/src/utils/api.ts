import type { OpenAIRequest, OpenAIResponse, OpenAIStreamChunk, EmotionContextPayload } from '../types/index';

/**
 * Safe API client for OpenAI integration
 * Handles authentication, streaming, and error handling
 */
export class OpenAIClient {
  private apiKey: string;
  private apiUrl: string;
  private model: string;
  private conversationId?: string;
  

  constructor(
    apiKey: string,
    apiUrl: string = 'https://api.openai.com/v1',
    model: string = 'gpt-4-turbo',
    conversationId?: string
  ) {
    this.apiKey = apiKey;
    this.apiUrl = apiUrl;
    this.model = model;
    this.conversationId = conversationId;
  }

  private async getErrorMessage(response: Response): Promise<string> {
    const text = await response.text();
    if (!text) return `HTTP ${response.status}`;

    try {
      const error = JSON.parse(text);
      return error.error?.message || error.message || text;
    } catch {
      return text;
    }
  }

  /**
   * Stream chat completion from OpenAI
   */
  async *streamChat(
    messages: Array<{ role: string; content: string }>,
    emotionContext?: EmotionContextPayload,
  ) {
    const baseMessages = messages.map(m => ({ role: m.role as 'user' | 'assistant' | 'system', content: m.content }));

    const request: OpenAIRequest = {
      model: this.model,
      messages: baseMessages,
      temperature: 0.7,
      stream: true,
      ...(this.conversationId ? { conversation_id: this.conversationId } : {}),
      ...(emotionContext ? { emotion_context: emotionContext } : {}),
    };

    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (this.apiKey && this.apiKey.trim() !== '') {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }

      const response = await fetch(`${this.apiUrl}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify(request),
        credentials: 'include',
      });

      if (!response.ok) {
        const errorMessage = await this.getErrorMessage(response);
        throw new Error(`API Error (${response.status}): ${errorMessage}`);
      }

      if (!response.body) {
        throw new Error('No response body from API');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              break;
            }
            try {
              const chunk = JSON.parse(data) as OpenAIStreamChunk;
              const content = chunk.choices[0]?.delta?.content;
              const reasoning = chunk.choices[0]?.delta?.reasoning_content;
              const toolExecution = chunk.choices[0]?.delta?.tool_execution;
              const toolCalls = chunk.choices[0]?.delta?.tool_calls;
              const emotion = chunk.choices[0]?.delta?.emotion;
              if (content || reasoning || toolExecution || toolCalls || emotion) {
                yield { content, reasoning, toolExecution, toolCalls, emotion };
              }
            } catch (e) {
              // Skip invalid JSON lines
              continue;
            }
          }
        }
      }
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Unknown API error occurred');
    }
  }

  /**
   * Get non-streaming chat completion
   */
  async chat(messages: Array<{ role: string; content: string }>): Promise<string> {
    const baseMessages = messages.map(m => ({ role: m.role as 'user' | 'assistant' | 'system', content: m.content }));

    const request: OpenAIRequest = {
      model: this.model,
      messages: baseMessages,
      temperature: 0.7,
      stream: false,
      ...(this.conversationId ? { conversation_id: this.conversationId } : {}),
    };

    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (this.apiKey && this.apiKey.trim() !== '') {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }

      const response = await fetch(`${this.apiUrl}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify(request),
        credentials: 'include',
      });

      if (!response.ok) {
        const errorMessage = await this.getErrorMessage(response);
        throw new Error(`API Error (${response.status}): ${errorMessage}`);
      }

      const data = (await response.json()) as OpenAIResponse;
      return data.choices[0]?.message?.content || 'No response received';
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Unknown API error occurred');
    }
  }
}
