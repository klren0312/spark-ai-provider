import { LanguageModelV1 } from '@ai-sdk/provider';
import {
  OpenAICompatibleChatLanguageModel,
  OpenAICompatibleCompletionLanguageModel,
} from '@ai-sdk/openai-compatible';
import {
  FetchFunction,
  loadApiKey,
  withoutTrailingSlash,
} from '@ai-sdk/provider-utils';
import {
  SparkChatModelId,
  SparkChatSettings,
} from './spark-chat-settings';

import {
  SparkCompletionModelId,
  SparkCompletionSettings,
} from './spark-completion-settings';

export interface SparkProviderSettings {
  /**
Spark API key.
*/
  apiKey?: string;
  /**
Base URL for the API calls.
*/
  baseURL?: string;
  /**
Custom headers to include in the requests.
*/
  headers?: Record<string, string>;
  /**
Custom fetch implementation. You can use it as a middleware to intercept requests,
or to provide a custom fetch implementation for e.g. testing.
*/
  fetch?: FetchFunction;
}

export interface SparkProvider {
  /**
Creates a model for text generation.
*/
  (
    modelId: SparkChatModelId,
    settings?: SparkChatSettings,
  ): LanguageModelV1;

  /**
Creates a chat model for text generation.
*/
  chatModel(
    modelId: SparkChatModelId,
    settings?: SparkChatSettings,
  ): LanguageModelV1;

  /**
Creates a completion model for text generation.
*/
  completionModel(
    modelId: SparkCompletionModelId,
    settings?: SparkCompletionSettings,
  ): LanguageModelV1;

}

export function createSparkProvider(
  options: SparkProviderSettings = {},
): SparkProvider {
  const baseURL = withoutTrailingSlash(
    options.baseURL ?? 'https://spark-api-open.xf-yun.com/v1',
  );
  const getHeaders = () => ({
    Authorization: `Bearer ${loadApiKey({
      apiKey: options.apiKey,
      environmentVariableName: 'SPARK_API_KEY',
      description: "Spark's API key",
    })}`,
    ...options.headers,
  });

  interface CommonModelConfig {
    provider: string;
    url: ({ path }: { path: string }) => string;
    headers: () => Record<string, string>;
    fetch?: FetchFunction;
  }

  const getCommonModelConfig = (modelType: string): CommonModelConfig => ({
    provider: `spark.${modelType}`,
    url: ({ path }) => `${baseURL}${path}`,
    headers: getHeaders,
    fetch: options.fetch,
  });

  const createChatModel = (
    modelId: SparkChatModelId,
    settings: SparkChatSettings = {},
  ) => {
    return new OpenAICompatibleChatLanguageModel(modelId, settings, {
      ...getCommonModelConfig('chat'),
      defaultObjectGenerationMode: 'json',
    });
  };

  const createCompletionModel = (
    modelId: SparkCompletionModelId,
    settings: SparkCompletionSettings = {},
  ) =>
    new OpenAICompatibleCompletionLanguageModel(
      modelId,
      settings,
      getCommonModelConfig('completion'),
    );


  const provider = (
    modelId: SparkChatModelId,
    settings?: SparkChatSettings,
  ) => createChatModel(modelId, settings);

  provider.completionModel = createCompletionModel;
  provider.chatModel = createChatModel;

  return provider as SparkProvider;
}

export const spark = createSparkProvider();
