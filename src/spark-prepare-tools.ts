import type {
  LanguageModelV1,
  LanguageModelV1CallWarning,
} from "@ai-sdk/provider"
import {
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider"

/**
 * @deprecated
 * mode will be removed in v2. All necessary settings will be directly supported 
 * through the call settings, in particular responseFormat, toolChoice, and tools.
 */
export function prepareTools({
  mode,
}: {
  mode: Parameters<LanguageModelV1["doGenerate"]>[0]["mode"] & {
    type: "regular"
  }
  structuredOutputs: boolean
}): {
    tools:
      | undefined
      | Array<{
        type: "function"
        function: {
          name: string
          description: string | undefined
          parameters: unknown
        }
      }>
    tool_choice:
      | { type: "function", function: { name: string } }
      | "auto"
      | "none"
      | "required"
      | undefined
    toolWarnings: LanguageModelV1CallWarning[]
  } {
  // Normalize tools array by converting empty array to undefined.
  const tools = mode.tools?.length ? mode.tools : undefined
  const toolWarnings: LanguageModelV1CallWarning[] = []

  if (tools == null) {
    return { tools: undefined, tool_choice: undefined, toolWarnings }
  }

  const toolChoice = mode.toolChoice
  const sparkCompatTools: Array<{
    type: "function"
    function: {
      name: string
      description: string | undefined
      parameters: unknown
    }
  }> = []

  // Process each tool and format for compatibility.
  for (const tool of tools) {
    if (tool.type === "provider-defined") {
      // Warn if the tool is provider-defined.
      toolWarnings.push({ type: "unsupported-tool", tool })
    }
    else {
      sparkCompatTools.push({
        type: "function",
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
      })
    }
  }

  if (toolChoice == null) {
    return { tools: sparkCompatTools, tool_choice: undefined, toolWarnings }
  }

  const type = toolChoice.type

  // Determine tool choice strategy.
  switch (type) {
    case "auto":
    case "none":
    case "required":
      return { tools: sparkCompatTools, tool_choice: type, toolWarnings }
    case "tool":
      return {
        tools: sparkCompatTools,
        tool_choice: {
          type: "function",
          function: {
            name: toolChoice.toolName,
          },
        },
        toolWarnings,
      }
    default: {
      // Exhaustive check to ensure all cases are handled.
      const _exhaustiveCheck: never = type
      throw new UnsupportedFunctionalityError({
        functionality: `Unsupported tool choice type: ${_exhaustiveCheck}`,
      })
    }
  }
}