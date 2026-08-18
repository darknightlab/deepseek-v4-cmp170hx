#!/usr/bin/env node

import { writeFile } from "node:fs/promises";

const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";
const DEFAULT_BENCHMARK_SUFFIX =
  "\nBased on the words above, write a short philosophical essay discussing " +
  "the meaning of existence, the nature of consciousness, and humanity's " +
  "place in the universe. Use clear, coherent sentences.";
const ASCII_WORDS = [
  "a",
  "the",
  "and",
  "of",
  "to",
  "in",
  "is",
  "it",
  "that",
  "for",
  "with",
  "as",
  "on",
  "by",
  "from",
  "this",
  "be",
  "are",
  "or",
  "not",
  "we",
  "you",
  "they",
  "can",
  "will",
  "if",
  "all",
  "one",
  "time",
  "world",
  "life",
  "work",
  "data",
  "model",
  "token",
  "text",
  "idea",
  "mind",
  "story",
  "light",
  "space",
  "future",
  "human",
  "system",
  "simple",
  "clear",
  "reason",
  "change",
  "value",
  "truth",
];
const LENGTHS = [512, 1024, 2048, 4096, 8192, 16384];
const SHORT_PROMPT_MAX_LENGTH = 46;
const BENCHMARK_SUFFIX_TOKEN_LENGTH = SHORT_PROMPT_MAX_LENGTH + 1;
const TOKEN_SANITY_MIN_TOKENS = 128;
const TOKEN_SANITY_MAX_RELATIVE_DIFF = 0.8;

function usage(exitCode = 0) {
  console.log(`Usage: node bench/prefill_speedtest.mjs [options]

OpenAI-compatible prefill benchmark derived from llm_speedtest/index.html.
The default input lengths are: ${LENGTHS.join(", ")}

Options:
  --url URL                 Chat completions endpoint
                            (default: $LLM_API_URL or
                             http://192.168.2.16:8000/v1/chat/completions)
  --model MODEL             Model name (default: $LLM_MODEL or
                             deepseek-ai/DeepSeek-V4-Flash-0731)
  --api-key KEY             API key (default: $OPENAI_API_KEY)
  --output-length N         Requested output tokens (default: 128)
  --concurrency N           Simultaneous requests per length (default: 1)
  --timeout MS              Request timeout per attempt (default: 30000)
  --temperature N           Sampling temperature (default: 1.0)
  --top-p N                 Top-p (default: 0.1)
  --presence-penalty N      Presence penalty (default: 0.0)
  --frequency-penalty N     Frequency penalty (default: 0.0)
  --retries N               Fetch attempts (default: 3)
  --retry-delay MS          Delay between retries (default: 1500)
  --json PATH               Also write full results as JSON
  --no-warmup               Skip the 96-token calibration request
  --no-latency-probe        Do not subtract /v1/models network latency
  -h, --help                Show this help
`);
  process.exit(exitCode);
}

function parseNumber(value, name, { integer = false, min = -Infinity } = {}) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < min || (integer && !Number.isInteger(number))) {
    throw new Error(`Invalid ${name}: ${value}`);
  }
  return number;
}

function parseArgs(argv) {
  const options = {
    url: process.env.LLM_API_URL ||
      "http://192.168.2.16:8000/v1/chat/completions",
    model: process.env.LLM_MODEL || "deepseek-ai/DeepSeek-V4-Flash-0731",
    apiKey: process.env.OPENAI_API_KEY || "",
    outputLength: 128,
    concurrency: 1,
    timeout: 30_000,
    temperature: 1.0,
    topP: 0.1,
    presencePenalty: 0.0,
    frequencyPenalty: 0.0,
    retries: 3,
    retryDelay: 1_500,
    jsonPath: null,
    warmup: true,
    latencyProbe: true,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = () => {
      index += 1;
      if (index >= argv.length) throw new Error(`Missing value for ${arg}`);
      return argv[index];
    };

    switch (arg) {
      case "--url": options.url = next(); break;
      case "--model": options.model = next(); break;
      case "--api-key": options.apiKey = next(); break;
      case "--output-length":
        options.outputLength = parseNumber(next(), arg, { integer: true, min: 1 });
        break;
      case "--concurrency":
        options.concurrency = parseNumber(next(), arg, { integer: true, min: 1 });
        break;
      case "--timeout":
        options.timeout = parseNumber(next(), arg, { integer: true, min: 1_000 });
        break;
      case "--temperature": options.temperature = parseNumber(next(), arg); break;
      case "--top-p": options.topP = parseNumber(next(), arg, { min: 0 }); break;
      case "--presence-penalty": options.presencePenalty = parseNumber(next(), arg); break;
      case "--frequency-penalty": options.frequencyPenalty = parseNumber(next(), arg); break;
      case "--retries":
        options.retries = parseNumber(next(), arg, { integer: true, min: 1 });
        break;
      case "--retry-delay":
        options.retryDelay = parseNumber(next(), arg, { integer: true, min: 0 });
        break;
      case "--json": options.jsonPath = next(); break;
      case "--no-warmup": options.warmup = false; break;
      case "--no-latency-probe": options.latencyProbe = false; break;
      case "-h":
      case "--help": usage(0); break;
      default: throw new Error(`Unknown option: ${arg}`);
    }
  }

  options.url = options.url.replace(/\/+$/, "");
  return options;
}

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function hasPositiveNumber(value) {
  return Number.isFinite(value) && value > 0;
}

function estimateTokenCount(text) {
  if (!text) return 0;
  let tokenCount = 0;
  let index = 0;
  const isCjk = (char) => /[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/.test(char);
  const isAsciiWord = (char) => /^[A-Za-z0-9_]$/.test(char);

  while (index < text.length) {
    const char = text[index];
    if (/\s/.test(char)) {
      index += 1;
    } else if (isCjk(char)) {
      tokenCount += 1;
      index += 1;
    } else if (isAsciiWord(char)) {
      const start = index;
      while (index < text.length && isAsciiWord(text[index])) index += 1;
      tokenCount += Math.max(1, Math.ceil((index - start) / 12));
    } else {
      tokenCount += 1;
      index += 1;
    }
  }
  return Math.max(1, tokenCount);
}

const PROMPT_TOKEN_WORDS = ASCII_WORDS.filter((word) => estimateTokenCount(word) === 1);

function estimatePromptTokens(prompt) {
  return estimateTokenCount(DEFAULT_SYSTEM_PROMPT) + estimateTokenCount(prompt);
}

function seededRandom(seed) {
  let state = (Date.now() ^ ((seed + 1) * 1_000_003)) >>> 0;
  return () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return state / 4_294_967_296;
  };
}

function buildTokenWordSequence(tokenCount, seed) {
  const count = Math.max(Math.trunc(Number(tokenCount) || 0), 0);
  const random = seededRandom(seed);
  const words = [];
  for (let index = 0; index < count; index += 1) {
    words.push(PROMPT_TOKEN_WORDS[Math.floor(random() * PROMPT_TOKEN_WORDS.length)]);
  }
  return words.join(" ");
}

function generatePrompt(length, seed) {
  const target = Math.max(Math.trunc(Number(length) || 0), 1);
  if (target <= SHORT_PROMPT_MAX_LENGTH) return buildTokenWordSequence(target, seed);
  const suffix = DEFAULT_BENCHMARK_SUFFIX.trim();
  const prefixTokens = Math.max(target - estimateTokenCount(suffix), 0);
  const prefix = buildTokenWordSequence(prefixTokens, seed);
  return prefix ? `${prefix}\n${suffix}` : suffix;
}

function normalizeUsage(usage = {}) {
  const values = (...candidates) => {
    const parsed = candidates
      .filter((value) => value !== null && value !== undefined && value !== "" && typeof value !== "boolean")
      .map(Number)
      .filter(Number.isFinite)
      .map(Math.trunc);
    const positive = parsed.filter((value) => value > 0);
    return positive.length ? Math.max(...positive) : (parsed.length ? Math.max(...parsed) : null);
  };

  const promptTokens = values(usage.prompt_tokens, usage.input_tokens, usage.prompt_eval_count);
  const completionTokens = values(usage.completion_tokens, usage.output_tokens, usage.eval_count);
  const reasoningTop = values(usage.reasoning_tokens);
  const reasoningNested = values(
    usage.completion_tokens_details?.reasoning_tokens,
    usage.output_tokens_details?.reasoning_tokens,
  );
  const totalTokens = values(usage.total_tokens);
  const reasoningTokens = reasoningTop ?? reasoningNested;
  let outputTokens = completionTokens;

  if (completionTokens !== null && reasoningTop !== null && reasoningTop > 0) {
    if (promptTokens !== null && totalTokens === promptTokens + completionTokens) {
      outputTokens = completionTokens;
    } else if (promptTokens !== null && totalTokens === promptTokens + completionTokens + reasoningTop) {
      outputTokens = completionTokens + reasoningTop;
    } else {
      outputTokens = completionTokens + reasoningTop;
    }
  } else if (outputTokens === null && reasoningTokens !== null) {
    outputTokens = reasoningTokens;
  } else if (outputTokens === null && promptTokens !== null && totalTokens !== null) {
    outputTokens = totalTokens - promptTokens;
  }

  return { promptTokens, completionTokens, reasoningTokens, outputTokens, totalTokens };
}

function resolvePromptTokens(apiTokens, estimate) {
  if (!hasPositiveNumber(apiTokens)) return { tokens: estimate, source: "local" };
  if (hasPositiveNumber(estimate)) {
    const relativeDiff = Math.abs(apiTokens - estimate) / Math.max(estimate, 1);
    if (Math.max(apiTokens, estimate) >= TOKEN_SANITY_MIN_TOKENS &&
        relativeDiff >= TOKEN_SANITY_MAX_RELATIVE_DIFF) {
      return { tokens: estimate, source: "local_api_anomaly", apiTokens, relativeDiff };
    }
  }
  return { tokens: apiTokens, source: "api" };
}

function buildCalibration(prompt, actualPromptTokens) {
  if (!hasPositiveNumber(actualPromptTokens)) return null;
  const estimatedPromptTokens = estimatePromptTokens(prompt);
  return {
    estimatedPromptTokens,
    actualPromptTokens,
    tokenOffset: actualPromptTokens - estimatedPromptTokens,
    tokenRatio: actualPromptTokens / Math.max(estimatedPromptTokens, 1),
  };
}

function calibratedGenerationLength(requestedLength, calibration) {
  if (!calibration || requestedLength <= SHORT_PROMPT_MAX_LENGTH) return requestedLength;
  const systemTokens = estimateTokenCount(DEFAULT_SYSTEM_PROMPT);
  return Math.max(
    BENCHMARK_SUFFIX_TOKEN_LENGTH,
    Math.round(requestedLength - systemTokens - calibration.tokenOffset),
  );
}

function calibratedEstimate(estimate, calibration) {
  if (!calibration) return Math.max(Math.trunc(estimate), 1);
  return Math.max(Math.round(estimate + calibration.tokenOffset), 1);
}

function modelListUrl(chatUrl) {
  const marker = "/v1/chat/completions";
  const index = chatUrl.indexOf(marker);
  if (index >= 0) return `${chatUrl.slice(0, index)}/v1/models`;
  return new URL("/v1/models", chatUrl).toString();
}

function headers(options) {
  const result = { "Content-Type": "application/json" };
  if (options.apiKey) result.Authorization = `Bearer ${options.apiKey}`;
  return result;
}

async function fetchWithRetry(url, fetchOptions, options) {
  let lastError;
  for (let attempt = 1; attempt <= options.retries; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), options.timeout);
    try {
      const response = await fetch(url, { ...fetchOptions, signal: controller.signal });
      clearTimeout(timeout);
      return response;
    } catch (error) {
      clearTimeout(timeout);
      lastError = error;
      if (attempt === options.retries) break;
      console.error(`  attempt ${attempt}/${options.retries} failed: ${error.message}; retrying`);
      await sleep(options.retryDelay);
    }
  }
  throw lastError;
}

async function warmup(options) {
  const prompt = generatePrompt(96, Math.floor(Math.random() * 1_000_000) + 1);
  const response = await fetchWithRetry(options.url, {
    method: "POST",
    headers: headers(options),
    body: JSON.stringify({
      model: options.model,
      messages: [
        { role: "system", content: DEFAULT_SYSTEM_PROMPT },
        { role: "user", content: prompt },
      ],
      max_tokens: 8,
      stream: false,
    }),
  }, { ...options, timeout: Math.max(options.timeout, 15_000) });
  if (!response.ok) throw new Error(`Warmup HTTP ${response.status}: ${await response.text()}`);
  const data = await response.json();
  const calibration = buildCalibration(prompt, normalizeUsage(data.usage).promptTokens);
  await sleep(800);
  return calibration;
}

async function networkLatency(options) {
  const started = performance.now();
  const response = await fetchWithRetry(modelListUrl(options.url), {
    method: "GET",
    headers: options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {},
    cache: "no-store",
  }, { ...options, timeout: Math.min(Math.max(options.timeout, 3_000), 8_000) });
  await response.text();
  return performance.now() - started;
}

function averageLatency(samples) {
  let values = samples.filter((value) => Number.isFinite(value) && value > 0);
  if (!values.length) return 0;
  if (values.length >= 3) values = [...values].sort((a, b) => a - b).slice(1, -1);
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function extractText(data) {
  if (data?.response) return { text: data.response, reasoning: false };
  const choice = data?.choices?.[0];
  if (choice?.delta?.reasoning_content) return { text: choice.delta.reasoning_content, reasoning: true };
  if (choice?.delta?.reasoning) return { text: choice.delta.reasoning, reasoning: true };
  if (choice?.delta?.content) return { text: choice.delta.content, reasoning: false };
  if (choice?.message?.content) return { text: choice.message.content, reasoning: false };
  if (choice?.text) return { text: choice.text, reasoning: false };
  if (data?.message?.thinking) return { text: data.message.thinking, reasoning: true };
  if (data?.message?.content) return { text: data.message.content, reasoning: false };
  return null;
}

async function measureOne(options, requestedLength, seed, calibration, latencyMs) {
  const generatedLength = calibratedGenerationLength(requestedLength, calibration);
  const prompt = generatePrompt(generatedLength, seed);
  const localPromptEstimate = calibratedEstimate(estimatePromptTokens(prompt), calibration);
  const request = {
    model: options.model,
    messages: [
      { role: "system", content: DEFAULT_SYSTEM_PROMPT },
      { role: "user", content: prompt },
    ],
    max_tokens: options.outputLength,
    temperature: options.temperature,
    top_p: options.topP,
    presence_penalty: options.presencePenalty,
    frequency_penalty: options.frequencyPenalty,
    stream: true,
    stream_options: { include_usage: true },
  };

  const startTime = performance.now();
  const response = await fetchWithRetry(options.url, {
    method: "POST",
    headers: headers(options),
    body: JSON.stringify(request),
  }, options);
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  if (!response.body) throw new Error("Response has no streaming body");

  let firstTokenTime = null;
  let endTime = null;
  let usage = null;
  let reasoning = "";
  let content = "";
  let pending = "";
  const decoder = new TextDecoder();

  const processLine = (line) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith(":")) return;
    const payload = trimmed.startsWith("data:") ? trimmed.slice(5).trim() : trimmed;
    if (!payload || payload === "[DONE]") return;
    let data;
    try {
      data = JSON.parse(payload);
    } catch {
      return;
    }
    if (data.usage) usage = data.usage;
    const part = extractText(data);
    if (part?.text) {
      if (part.reasoning) reasoning += part.text;
      else content += part.text;
      if (firstTokenTime === null) firstTokenTime = performance.now();
    }
  };

  for await (const bytes of response.body) {
    pending += decoder.decode(bytes, { stream: true });
    const lines = pending.split(/\r?\n/);
    pending = lines.pop() ?? "";
    for (const line of lines) processLine(line);
  }
  pending += decoder.decode();
  for (const line of pending.split(/\r?\n/)) processLine(line);
  endTime = performance.now();

  if (firstTokenTime === null) throw new Error("Stream ended without a reasoning/content token");
  const usageStats = normalizeUsage(usage);
  const promptResolution = resolvePromptTokens(usageStats.promptTokens, localPromptEstimate);
  const clientPrefillMs = Math.max((firstTokenTime - startTime) - latencyMs, 1);
  const clientDecodeMs = Math.max(endTime - firstTokenTime, 1);
  const outputTokens = usageStats.outputTokens ??
    (estimateTokenCount(reasoning) + estimateTokenCount(content));

  return {
    requestedPromptTokens: requestedLength,
    actualPromptTokens: promptResolution.tokens,
    apiPromptTokens: usageStats.promptTokens,
    promptTokenSource: promptResolution.source,
    cachedPromptTokens: usage?.prompt_tokens_details?.cached_tokens ?? null,
    outputTokens,
    ttftMs: firstTokenTime - startTime,
    prefillMs: clientPrefillMs,
    prefillTokensPerSecond: promptResolution.tokens / (clientPrefillMs / 1000),
    decodeMs: clientDecodeMs,
    decodeTokensPerSecond: outputTokens / (clientDecodeMs / 1000),
    networkLatencyMs: latencyMs,
    startTimestamp: startTime,
    firstTokenTimestamp: firstTokenTime,
    endTimestamp: endTime,
    reasoningChars: reasoning.length,
    contentChars: content.length,
  };
}

function aggregate(results, latencyMs) {
  const totalPromptTokens = results.reduce((sum, result) => sum + result.actualPromptTokens, 0);
  const totalOutputTokens = results.reduce((sum, result) => sum + result.outputTokens, 0);
  const minStart = Math.min(...results.map((result) => result.startTimestamp));
  const maxFirst = Math.max(...results.map((result) => result.firstTokenTimestamp));
  const minFirst = Math.min(...results.map((result) => result.firstTokenTimestamp));
  const maxEnd = Math.max(...results.map((result) => result.endTimestamp));
  const prefillMs = Math.max((maxFirst - minStart) - latencyMs, 1);
  const decodeMs = Math.max(maxEnd - minFirst, 1);
  return {
    totalPromptTokens,
    totalOutputTokens,
    prefillMs,
    decodeMs,
    prefillTokensPerSecond: totalPromptTokens / (prefillMs / 1000),
    decodeTokensPerSecond: totalOutputTokens / (decodeMs / 1000),
  };
}

async function main() {
  let options;
  try {
    options = parseArgs(process.argv.slice(2));
  } catch (error) {
    console.error(error.message);
    usage(2);
  }

  console.log(`Endpoint:    ${options.url}`);
  console.log(`Model:       ${options.model}`);
  console.log(`Lengths:     ${LENGTHS.join(", ")}`);
  console.log(`Output:      ${options.outputLength}`);
  console.log(`Concurrency: ${options.concurrency}`);
  console.log(`Timing:      client TTFT${options.latencyProbe ? " minus /v1/models latency" : ""}`);
  console.log();

  let calibration = null;
  if (options.warmup) {
    process.stdout.write("Warmup/calibration... ");
    calibration = await warmup(options);
    if (calibration) {
      console.log(
        `OK (estimated=${calibration.estimatedPromptTokens}, ` +
        `actual=${calibration.actualPromptTokens}, offset=${calibration.tokenOffset})`,
      );
    } else {
      console.log("completed without usage calibration");
    }
  }

  const latencySamples = [];
  if (options.latencyProbe) {
    for (let index = 0; index < 2; index += 1) {
      latencySamples.push(await networkLatency(options));
    }
    console.log(`Initial network latency: ${averageLatency(latencySamples).toFixed(2)} ms`);
  }

  const allResults = [];
  console.log();
  console.log("requested  actual  prefill_ms  prefill_tok/s  ttft_ms  cached  ok");
  console.log("---------  ------  ----------  -------------  -------  ------  --");

  for (const length of LENGTHS) {
    if (options.latencyProbe) latencySamples.push(await networkLatency(options));
    const latencyMs = options.latencyProbe ? averageLatency(latencySamples) : 0;
    const promises = [];
    for (let index = 0; index < options.concurrency; index += 1) {
      promises.push(measureOne(options, length, index + 1, calibration, latencyMs));
    }
    const settled = await Promise.allSettled(promises);
    const successes = settled.filter((result) => result.status === "fulfilled").map((result) => result.value);
    const failures = settled.filter((result) => result.status === "rejected");
    if (!successes.length) {
      console.log(`${String(length).padStart(9)}  FAILED: ${failures.map((failure) => failure.reason.message).join("; ")}`);
      allResults.push({ requestedPromptTokens: length, error: failures.map((failure) => failure.reason.message).join("; ") });
    } else {
      const combined = aggregate(successes, latencyMs);
      const actualAverage = successes.reduce((sum, result) => sum + result.actualPromptTokens, 0) / successes.length;
      const ttftAverage = successes.reduce((sum, result) => sum + result.ttftMs, 0) / successes.length;
      const cached = successes.reduce((sum, result) => sum + (result.cachedPromptTokens || 0), 0);
      console.log(
        `${String(length).padStart(9)}  ` +
        `${actualAverage.toFixed(0).padStart(6)}  ` +
        `${combined.prefillMs.toFixed(2).padStart(10)}  ` +
        `${combined.prefillTokensPerSecond.toFixed(2).padStart(13)}  ` +
        `${ttftAverage.toFixed(2).padStart(7)}  ` +
        `${String(cached).padStart(6)}  ` +
        `${successes.length}/${options.concurrency}`,
      );
      allResults.push({
        requestedPromptTokens: length,
        actualPromptTokensAverage: actualAverage,
        networkLatencyMs: latencyMs,
        ...combined,
        successful: successes.length,
        concurrency: options.concurrency,
        requests: successes,
        errors: failures.map((failure) => failure.reason.message),
      });
    }
    await sleep(1_500);
  }

  const output = {
    timestamp: new Date().toISOString(),
    source: "https://gengchaogit.github.io/llm_speedtest/",
    options: { ...options, apiKey: options.apiKey ? "[redacted]" : "" },
    calibration,
    results: allResults,
  };

  if (options.jsonPath) {
    await writeFile(options.jsonPath, `${JSON.stringify(output, null, 2)}\n`, "utf8");
    console.log(`\nWrote ${options.jsonPath}`);
  }
}

main().catch((error) => {
  console.error(`Fatal: ${error.stack || error.message}`);
  process.exitCode = 1;
});
