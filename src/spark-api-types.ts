import type { JSONValue } from "@ai-sdk/provider"

/**
 * Type representing a prompt for Spark Chat built from an array of messages.
 */
export type SparkChatPrompt = Array<SparkMessage>

/**
 * Union type for all possible Spark message types.
 */
export type SparkMessage =
  | SparkSystemMessage
  | SparkUserMessage
  | SparkAssistantMessage
  | SparkToolMessage

// Helper type for additional properties for metadata extensibility.
type JsonRecord<T = never> = Record<
  string,
  JSONValue | JSONValue[] | T | T[] | undefined
>

/**
 * System messages contain instructions/data set by the system.
 */
export interface SparkSystemMessage extends JsonRecord {
  role: "system"
  content: string
}

/**
 * User messages sent by the user to the model.
 */
export interface SparkUserMessage
  extends JsonRecord<SparkContentPart> {
  role: "user"
  content: string | Array<SparkContentPart>
}

/**
 * Represents a part of a message content.
 */
export type SparkContentPart =
  | SparkContentPartText
  | SparkContentPartImage

/**
 * Message part that contains an image URL.
 */
export interface SparkContentPartImage extends JsonRecord {
  type: "image_url"
  image_url: { url: string }
}

/**
 * Message part that contains text.
 */
export interface SparkContentPartText extends JsonRecord {
  type: "text"
  text: string
}

/**
 * Assistant messages response from the model.
 */
export interface SparkAssistantMessage
  extends JsonRecord<SparkMessageToolCall> {
  role: "assistant"
  content?: string | null
  tool_calls?: SparkMessageToolCall
}

/**
 * Represents a tool call embedded within an assistant message.
 */
export interface SparkMessageToolCall extends JsonRecord {
  type: "function"
  function: {
    arguments: string
    name: string
  }
}

/**
 * Represents the response from a tool.
 */
export interface SparkToolMessage extends JsonRecord {
  role: "tool"
  content: string
  tool_call_id: string
}