import { OpenAICompatibleChatSettings } from '@ai-sdk/openai-compatible';

// https://xinghuo.xfyun.cn/spark
export type SparkChatModelId =
  | 'lite'
  | 'generalv3'
  | 'pro-128k'
  | 'generalv3.5'
  | 'max-32k'
  | '4.0Ultra'
  | (string & {});

export interface SparkChatSettings extends OpenAICompatibleChatSettings {}
