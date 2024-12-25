# AI SDK - Spark Provider

The **[Spark provider](https://github.com/klren0312/spark-ai-provider)** contains language model support for the Spark API, giving you access to models like lite, generalv3, pro-128k, generalv3.5, max-32k and 4.0Ultra.

## Setup

The Spark provider is available in the `spark-ai-provider` module. You can install it with

```bash
npm i spark-ai-provider
```

## Provider Instance

You can import `createSparkProvider` from `spark-ai-provider` to create a provider instance:

```ts
import { createSparkProvider } from 'spark-ai-provider';
```

## Example

```ts
import { createSparkProvider } from './index.mjs';
import { generateText } from 'ai';
const spark = createSparkProvider({
  apiKey: '',
});
const { text } = await generateText({
  model: spark('lite'),
  prompt: 'Write a vegetarian lasagna recipe for 4 people.',
});
```

## Documentation

Please check out the **[Spark provider documentation](https://github.com/klren0312/spark-ai-provider)** for more information.