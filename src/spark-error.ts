import type { ZodSchema } from "zod"
import { createJsonErrorResponseHandler } from "@ai-sdk/provider-utils"
import { z } from "zod"

/**
 * Schema defining the structure of a Spark error response.
 */
const sparkErrorDataSchema = z.object({
  object: z.literal("error"),
  message: z.string(),
  type: z.string(),
  param: z.string().nullable(),
  code: z.string().nullable(),
})

export type SparkErrorData = z.infer<typeof sparkErrorDataSchema>

/**
 * Interface for defining error structures for Spark.
 */
export interface SparkErrorStructure<T> {
  /**
   * Zod schema to validate error data.
   */
  errorSchema: ZodSchema<T>
  /**
   * Maps error details to a human-readable message.
   */
  errorToMessage: (error: T) => string
  /**
   * Determines if an error is retryable.
   */
  isRetryable?: (response: Response, error?: T) => boolean
}

// Create a handler for failed responses using the defined schema.
export const sparkFailedResponseHandler = createJsonErrorResponseHandler({
  errorSchema: sparkErrorDataSchema,
  errorToMessage: error => error.message,
})

export const defaultSparkErrorStructure: SparkErrorStructure<SparkErrorData> = {
  errorSchema: sparkErrorDataSchema,
  errorToMessage: data => data.message,
}