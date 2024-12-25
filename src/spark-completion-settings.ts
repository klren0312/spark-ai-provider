import { OpenAICompatibleCompletionSettings } from '@ai-sdk/openai-compatible';
import { SparkChatModelId } from './spark-chat-settings';

// Use the same model IDs as chat
export type SparkCompletionModelId = SparkChatModelId;

export interface SparkCompletionSettings
  extends OpenAICompatibleCompletionSettings {}
