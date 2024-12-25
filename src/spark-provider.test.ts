import { describe, it, expect, vi, beforeEach, Mock } from 'vitest';
import { createSparkProvider } from './spark-provider';
import { LanguageModelV1 } from '@ai-sdk/provider';
import { loadApiKey } from '@ai-sdk/provider-utils';
import {
  OpenAICompatibleChatLanguageModel,
  OpenAICompatibleCompletionLanguageModel,
} from '@ai-sdk/openai-compatible';

// Add type assertion for the mocked class
const OpenAICompatibleChatLanguageModelMock =
  OpenAICompatibleChatLanguageModel as unknown as Mock;

vi.mock('@ai-sdk/openai-compatible', () => ({
  OpenAICompatibleChatLanguageModel: vi.fn(),
  OpenAICompatibleCompletionLanguageModel: vi.fn(),
  OpenAICompatibleEmbeddingModel: vi.fn(),
}));

vi.mock('@ai-sdk/provider-utils', () => ({
  loadApiKey: vi.fn().mockReturnValue('mock-api-key'),
  withoutTrailingSlash: vi.fn(url => url),
}));

describe('SparkProvider', () => {
  let mockLanguageModel: LanguageModelV1;

  beforeEach(() => {
    // Mock implementations of models
    mockLanguageModel = {
      // Add any required methods for LanguageModelV1
    } as LanguageModelV1;

    // Reset mocks
    vi.clearAllMocks();
  });

  describe('createSparkProvider', () => {
    it('should create a SparkProvider instance with default options', () => {
      const provider = createSparkProvider();
      const model = provider('model-id');

      // Use the mocked version
      const constructorCall =
        OpenAICompatibleChatLanguageModelMock.mock.calls[0];
      const config = constructorCall[2];
      config.headers();

      expect(loadApiKey).toHaveBeenCalledWith({
        apiKey: undefined,
        environmentVariableName: 'SPARK_API_KEY',
        description: "Spark's API key",
      });
    });

    it('should create a SparkProvider instance with custom options', () => {
      const options = {
        apiKey: 'custom-key',
        baseURL: 'https://custom.url',
        headers: { 'Custom-Header': 'value' },
      };
      const provider = createSparkProvider(options);
      const model = provider('model-id');

      const constructorCall =
        OpenAICompatibleChatLanguageModelMock.mock.calls[0];
      const config = constructorCall[2];
      config.headers();

      expect(loadApiKey).toHaveBeenCalledWith({
        apiKey: 'custom-key',
        environmentVariableName: 'SPARK_API_KEY',
        description: "Spark's API key",
      });
    });

    it('should return a chat model when called as a function', () => {
      const provider = createSparkProvider();
      const modelId = 'foo-model-id';
      const settings = { user: 'foo-user' };

      const model = provider(modelId, settings);
      expect(model).toBeInstanceOf(OpenAICompatibleChatLanguageModel);
    });
  });

  describe('chatModel', () => {
    it('should construct a chat model with correct configuration', () => {
      const provider = createSparkProvider();
      const modelId = 'lite';
      const settings = { user: 'foo-user' };

      const model = provider.chatModel(modelId, settings);

      expect(model).toBeInstanceOf(OpenAICompatibleChatLanguageModel);
      expect(OpenAICompatibleChatLanguageModelMock).toHaveBeenCalledWith(
        modelId,
        settings,
        expect.objectContaining({
          provider: 'spark.chat',
          defaultObjectGenerationMode: 'json',
        }),
      );
    });
  });

  describe('completionModel', () => {
    it('should construct a completion model with correct configuration', () => {
      const provider = createSparkProvider();
      const modelId = 'spark-completion-model';
      const settings = { user: 'foo-user' };

      const model = provider.completionModel(modelId, settings);

      expect(model).toBeInstanceOf(OpenAICompatibleCompletionLanguageModel);
    });
  });
});
