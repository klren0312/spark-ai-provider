import type {
  LanguageModelV1Prompt,
  LanguageModelV1ProviderMetadata,
} from "@ai-sdk/provider"
import type { SparkChatPrompt } from "./spark-api-types"
import {
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider"
import { convertUint8ArrayToBase64 } from "@ai-sdk/provider-utils"

// JSDoc for helper function to extract Spark metadata.
/**
 * Extracts Spark-specific metadata from a message.
 *
 * @param message - An object that may contain providerMetadata
 * @param message.providerMetadata - Provider-specific metadata containing Spark configuration
 * @returns The Spark metadata object or an empty object if none exists
 */

function getSparkMetadata(message: {
  providerMetadata?: LanguageModelV1ProviderMetadata
}) {
  return message?.providerMetadata?.spark ?? {}
}

/**
 * Converts a generic language model prompt to Spark chat messages.
 *
 * @param prompt The language model prompt to convert.
 * @returns An array of Spark chat messages.
 */
export function convertToSparkChatMessages(
  prompt: LanguageModelV1Prompt,
): SparkChatPrompt {
  const messages: SparkChatPrompt = []
  // Iterate over each prompt message.
  for (const { role, content, ...message } of prompt) {
    const metadata = getSparkMetadata({ ...message })
    switch (role) {
      case "system": {
        // System messages are sent directly with metadata.
        messages.push({ role: "system", content, ...metadata })
        break
      }

      case "user": {
        if (content.length === 1 && content[0].type === "text") {
          // For a single text element, simplify the conversion.
          messages.push({
            role: "user",
            content: content[0].text,
            ...getSparkMetadata(content[0]),
          })
          break
        }
        // For multiple content parts, process each part.
        messages.push({
          role: "user",
          content: content.map((part) => {
            const partMetadata = getSparkMetadata(part)
            switch (part.type) {
              case "text": {
                // Plain text conversion.
                return { type: "text", text: part.text, ...partMetadata }
              }
              case "image": {
                // Convert images and encode if necessary.
                return {
                  type: "image_url",
                  image_url: {
                    url:
                      part.image instanceof URL
                        ? part.image.toString()
                        : `data:${
                          part.mimeType ?? "image/jpeg"
                        };base64,${convertUint8ArrayToBase64(part.image)}`,
                  },
                  ...partMetadata,
                }
              }
              default: {
                // Unsupported file content parts trigger an error.
                throw new UnsupportedFunctionalityError({
                  functionality: "File content parts in user messages",
                })
              }
            }
          }),
          ...metadata,
        })

        break
      }

      case "assistant": {
        // Build text response and accumulate function/tool calls.
        let text = ""
        let toolCalls: {
          type: 'function';
          function: {
            arguments: string;
            name: string;
          };
        } = {
          type: 'function',
          function: {
            arguments: '',
            name: '',
          },
        }

        for (const part of content) {
          const partMetadata = getSparkMetadata(part)
          switch (part.type) {
            case "text": {
              // Append each text part.
              text += part.text
              break
            }
            case "tool-call": {
              // Convert tool calls to function calls with serialized arguments.
              toolCalls = {
                type: "function",
                function: {
                  name: part.toolName,
                  arguments: JSON.stringify(part.args),
                },
                ...partMetadata,
              }
              break
            }
            default: {
              // This branch should never occur.
              const _exhaustiveCheck: any = part
              throw new Error(`Unsupported part: ${_exhaustiveCheck}`)
            }
          }
        }

        messages.push({
          role: "assistant",
          content: text,
          tool_calls: toolCalls ? toolCalls : undefined,
          ...metadata,
        })

        break
      }

      case "tool": {
        // Process tool responses by converting result to JSON string.
        for (const toolResponse of content) {
          const toolResponseMetadata = getSparkMetadata(toolResponse)
          messages.push({
            role: "tool",
            tool_call_id: toolResponse.toolCallId,
            content: JSON.stringify(toolResponse.result),
            ...toolResponseMetadata,
          })
        }
        break
      }

      default: {
        // Ensure all roles are handled.
        const _exhaustiveCheck: never = role
        throw new Error(`Unsupported role: ${_exhaustiveCheck}`)
      }
    }
  }

  return messages
}