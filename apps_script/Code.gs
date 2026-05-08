const PERPLEXITY_ENDPOINT = 'https://api.perplexity.ai/v1/agent';
const DEFAULT_PERPLEXITY_MODEL = 'openai/gpt-5.4';
const DEFAULT_MAX_OUTPUT_TOKENS = 500;
const DEFAULT_CACHE_SECONDS = 21600;
const SCRIPT_PROPERTY_API_KEY = 'PERPLEXITY_API_KEY';
const SCRIPT_PROPERTY_MODEL = 'PERPLEXITY_MODEL';
const SCRIPT_PROPERTY_PRESET = 'PERPLEXITY_PRESET';

/**
 * Calls Perplexity from a Google Sheets cell.
 *
 * @param {string} prompt Prompt to send to Perplexity.
 * @param {boolean=} useWebSearch Whether to enable web_search and fetch_url tools.
 * @return {string} Perplexity response text.
 * @customfunction
 */
function PERPLEXITY(prompt, useWebSearch) {
  const promptText = String(prompt || '').trim();
  if (!promptText) {
    throw new Error('PERPLEXITY requires a non-empty prompt.');
  }

  return callPerplexityText_(promptText, Boolean(useWebSearch));
}

/**
 * Uses headers, sample rows, and an input row to classify or extract one value.
 *
 * @param {Object[][]} headers Header row range.
 * @param {Object[][]} sampleRows Sample data rows that illustrate column meanings.
 * @param {Object[][]} inputRow Row to classify or extract from.
 * @param {string} instruction Natural-language task instruction.
 * @param {boolean=} useWebSearch Whether to enable web_search and fetch_url tools.
 * @return {string} The primary structured result.
 * @customfunction
 */
function PERPLEXITY_AUTO(headers, sampleRows, inputRow, instruction, useWebSearch) {
  const result = callPerplexityAuto_(headers, sampleRows, inputRow, instruction, Boolean(useWebSearch));
  return result.result;
}

/**
 * Returns the full auto-mode structured payload as JSON.
 *
 * @param {Object[][]} headers Header row range.
 * @param {Object[][]} sampleRows Sample data rows that illustrate column meanings.
 * @param {Object[][]} inputRow Row to classify or extract from.
 * @param {string} instruction Natural-language task instruction.
 * @param {boolean=} useWebSearch Whether to enable web_search and fetch_url tools.
 * @return {string} JSON containing result, confidence, and rationale.
 * @customfunction
 */
function PERPLEXITY_AUTO_JSON(headers, sampleRows, inputRow, instruction, useWebSearch) {
  const result = callPerplexityAuto_(headers, sampleRows, inputRow, instruction, Boolean(useWebSearch));
  return JSON.stringify(result);
}

/**
 * Adds a small setup menu when the script is bound to a spreadsheet.
 */
function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('csv-to-llm')
    .addItem('Set Perplexity API key', 'promptForPerplexityApiKey')
    .addItem('Set Perplexity model', 'promptForPerplexityModel')
    .addItem('Set Perplexity preset', 'promptForPerplexityPreset')
    .addToUi();
}

/**
 * Prompts for and stores the Perplexity API key in script properties.
 */
function promptForPerplexityApiKey() {
  const ui = SpreadsheetApp.getUi();
  const response = ui.prompt('Perplexity API key', 'Paste your Perplexity API key.', ui.ButtonSet.OK_CANCEL);
  if (response.getSelectedButton() !== ui.Button.OK) {
    return;
  }

  const apiKey = response.getResponseText().trim();
  if (!apiKey) {
    ui.alert('No API key was saved.');
    return;
  }

  PropertiesService.getScriptProperties().setProperty(SCRIPT_PROPERTY_API_KEY, apiKey);
  ui.alert('Perplexity API key saved.');
}

/**
 * Prompts for and stores the Agent API model name.
 */
function promptForPerplexityModel() {
  const ui = SpreadsheetApp.getUi();
  const currentModel = getScriptProperty_(SCRIPT_PROPERTY_MODEL) || DEFAULT_PERPLEXITY_MODEL;
  const response = ui.prompt(
    'Perplexity model',
    'Enter an Agent API model name. Current/default: ' + currentModel,
    ui.ButtonSet.OK_CANCEL
  );
  if (response.getSelectedButton() !== ui.Button.OK) {
    return;
  }

  const model = response.getResponseText().trim();
  if (!model) {
    ui.alert('No model was saved.');
    return;
  }

  const properties = PropertiesService.getScriptProperties();
  properties.setProperty(SCRIPT_PROPERTY_MODEL, model);
  properties.deleteProperty(SCRIPT_PROPERTY_PRESET);
  ui.alert('Perplexity model saved.');
}

/**
 * Prompts for and stores a Perplexity preset name.
 */
function promptForPerplexityPreset() {
  const ui = SpreadsheetApp.getUi();
  const response = ui.prompt(
    'Perplexity preset',
    'Enter a preset name such as pro-search. This clears any model override.',
    ui.ButtonSet.OK_CANCEL
  );
  if (response.getSelectedButton() !== ui.Button.OK) {
    return;
  }

  const preset = response.getResponseText().trim();
  if (!preset) {
    ui.alert('No preset was saved.');
    return;
  }

  setPerplexityPreset(preset);
  ui.alert('Perplexity preset saved.');
}

/**
 * Stores the Perplexity API key without using the spreadsheet UI.
 *
 * @param {string} apiKey Perplexity API key.
 */
function setPerplexityApiKey(apiKey) {
  const value = String(apiKey || '').trim();
  if (!value) {
    throw new Error('setPerplexityApiKey requires a non-empty API key.');
  }

  PropertiesService.getScriptProperties().setProperty(SCRIPT_PROPERTY_API_KEY, value);
}

/**
 * Stores the preferred Agent API model name.
 *
 * @param {string} model Agent API model name.
 */
function setPerplexityModel(model) {
  const value = String(model || '').trim();
  if (!value) {
    throw new Error('setPerplexityModel requires a non-empty model name.');
  }

  const properties = PropertiesService.getScriptProperties();
  properties.setProperty(SCRIPT_PROPERTY_MODEL, value);
  properties.deleteProperty(SCRIPT_PROPERTY_PRESET);
}

/**
 * Stores a Perplexity preset and clears any model override.
 *
 * @param {string} preset Perplexity preset name, such as pro-search.
 */
function setPerplexityPreset(preset) {
  const value = String(preset || '').trim();
  if (!value) {
    throw new Error('setPerplexityPreset requires a non-empty preset name.');
  }

  const properties = PropertiesService.getScriptProperties();
  properties.setProperty(SCRIPT_PROPERTY_PRESET, value);
  properties.deleteProperty(SCRIPT_PROPERTY_MODEL);
}

/**
 * Calls Perplexity and returns plain text.
 *
 * @param {string} promptText Prompt text.
 * @param {boolean} useWebSearch Whether web tools should be enabled.
 * @return {string} Response text.
 */
function callPerplexityText_(promptText, useWebSearch) {
  const payload = buildPerplexityPayload_(promptText, {
    useWebSearch: useWebSearch,
    instructions: buildTextInstructions_(useWebSearch),
    maxOutputTokens: DEFAULT_MAX_OUTPUT_TOKENS
  });

  const response = fetchPerplexityWithCache_(payload);
  const outputText = extractOutputText_(response);
  if (!outputText) {
    throw new Error('Perplexity returned an empty response.');
  }

  return outputText.trim();
}

/**
 * Calls Perplexity with a spreadsheet-aware structured prompt.
 *
 * @param {*} headers Header row range or value.
 * @param {*} sampleRows Sample row range or value.
 * @param {*} inputRow Input row range or value.
 * @param {string} instruction Task instruction.
 * @param {boolean} useWebSearch Whether web tools should be enabled.
 * @return {{result: string, confidence: string, rationale: string}} Parsed structured result.
 */
function callPerplexityAuto_(headers, sampleRows, inputRow, instruction, useWebSearch) {
  const instructionText = String(instruction || '').trim();
  if (!instructionText) {
    throw new Error('PERPLEXITY_AUTO requires a non-empty instruction.');
  }

  const headerValues = flattenRange_(headers).map(function (value, index) {
    const header = String(value || '').trim();
    return header || 'Column ' + (index + 1);
  });

  if (headerValues.length === 0) {
    throw new Error('PERPLEXITY_AUTO requires at least one header.');
  }

  const sampleObjects = rowsToObjects_(normalizeRange_(sampleRows), headerValues);
  const inputObject = rowToObject_(flattenRange_(inputRow), headerValues);
  const promptText = buildAutoPrompt_(instructionText, headerValues, sampleObjects, inputObject);
  const payload = buildPerplexityPayload_(promptText, {
    useWebSearch: useWebSearch,
    instructions: buildAutoInstructions_(useWebSearch),
    maxOutputTokens: DEFAULT_MAX_OUTPUT_TOKENS,
    responseFormat: buildAutoResponseFormat_()
  });

  const response = fetchPerplexityWithCache_(payload);
  return parseAutoResponse_(extractOutputText_(response));
}

/**
 * Builds the Agent API payload.
 *
 * @param {string} input Prompt input.
 * @param {{useWebSearch: boolean, instructions: string, maxOutputTokens: number, responseFormat: Object=}} options Call options.
 * @return {Object} Agent API payload.
 */
function buildPerplexityPayload_(input, options) {
  const payload = {
    input: input,
    instructions: options.instructions,
    max_output_tokens: options.maxOutputTokens
  };

  const preset = getScriptProperty_(SCRIPT_PROPERTY_PRESET);
  if (preset) {
    payload.preset = preset;
  } else {
    payload.model = getScriptProperty_(SCRIPT_PROPERTY_MODEL) || DEFAULT_PERPLEXITY_MODEL;
  }

  if (options.useWebSearch) {
    payload.tools = [
      { type: 'web_search' },
      { type: 'fetch_url' }
    ];
  }

  if (options.responseFormat) {
    payload.response_format = options.responseFormat;
  }

  return payload;
}

/**
 * Fetches Perplexity with CacheService-backed memoization.
 *
 * @param {Object} payload Agent API request payload.
 * @return {Object} Parsed response body.
 */
function fetchPerplexityWithCache_(payload) {
  const cache = CacheService.getScriptCache();
  const cacheKey = buildCacheKey_(payload);
  const cached = cache.get(cacheKey);
  if (cached) {
    return JSON.parse(cached);
  }

  const response = fetchPerplexity_(payload);
  cache.put(cacheKey, JSON.stringify(response), DEFAULT_CACHE_SECONDS);
  return response;
}

/**
 * Sends a Perplexity Agent API request.
 *
 * @param {Object} payload Agent API request payload.
 * @return {Object} Parsed response body.
 */
function fetchPerplexity_(payload) {
  const apiKey = getScriptProperty_(SCRIPT_PROPERTY_API_KEY);
  if (!apiKey) {
    throw new Error('Missing PERPLEXITY_API_KEY. Run promptForPerplexityApiKey or setPerplexityApiKey first.');
  }

  const response = UrlFetchApp.fetch(PERPLEXITY_ENDPOINT, {
    method: 'post',
    contentType: 'application/json',
    headers: {
      Authorization: 'Bearer ' + apiKey
    },
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  });

  const statusCode = response.getResponseCode();
  const bodyText = response.getContentText();
  if (statusCode < 200 || statusCode >= 300) {
    throw new Error('Perplexity API error ' + statusCode + ': ' + bodyText);
  }

  const parsedBody = JSON.parse(bodyText);
  if (parsedBody.status === 'failed') {
    const message = parsedBody.error && parsedBody.error.message ? parsedBody.error.message : bodyText;
    throw new Error('Perplexity API failed: ' + message);
  }

  return parsedBody;
}

/**
 * Extracts text from Agent API responses.
 *
 * @param {Object} response Parsed response body.
 * @return {string} Output text.
 */
function extractOutputText_(response) {
  if (response.output_text) {
    return String(response.output_text);
  }

  if (!Array.isArray(response.output)) {
    return '';
  }

  const chunks = [];
  response.output.forEach(function (item) {
    if (typeof item.content === 'string') {
      chunks.push(item.content);
      return;
    }

    if (!Array.isArray(item.content)) {
      return;
    }

    item.content.forEach(function (contentItem) {
      if (contentItem.text) {
        chunks.push(contentItem.text);
      }
    });
  });

  return chunks.join('\n').trim();
}

/**
 * Parses the JSON text returned by PERPLEXITY_AUTO.
 *
 * @param {string} outputText Raw output text.
 * @return {{result: string, confidence: string, rationale: string}} Normalized result.
 */
function parseAutoResponse_(outputText) {
  if (!outputText) {
    throw new Error('Perplexity returned an empty auto-mode response.');
  }

  const parsed = JSON.parse(outputText);
  if (!parsed.result) {
    throw new Error('Perplexity auto-mode response did not include result.');
  }

  return {
    result: String(parsed.result),
    confidence: parsed.confidence ? String(parsed.confidence) : '',
    rationale: parsed.rationale ? String(parsed.rationale) : ''
  };
}

/**
 * Builds the JSON Schema response format for PERPLEXITY_AUTO.
 *
 * @return {Object} JSON Schema response format.
 */
function buildAutoResponseFormat_() {
  return {
    type: 'json_schema',
    json_schema: {
      name: 'SheetsAutoResult',
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['result', 'confidence', 'rationale'],
        properties: {
          result: {
            type: 'string',
            description: 'The single spreadsheet-cell answer requested by the user.'
          },
          confidence: {
            type: 'string',
            description: 'A concise confidence label such as high, medium, or low.'
          },
          rationale: {
            type: 'string',
            description: 'One short sentence explaining the result.'
          }
        }
      }
    }
  };
}

/**
 * Builds a spreadsheet-aware prompt for auto mode.
 *
 * @param {string} instruction User instruction.
 * @param {string[]} headers Header names.
 * @param {Object[]} sampleObjects Sample row objects.
 * @param {Object} inputObject Target row object.
 * @return {string} Prompt text.
 */
function buildAutoPrompt_(instruction, headers, sampleObjects, inputObject) {
  return [
    'Task:',
    instruction,
    '',
    'Column headers:',
    JSON.stringify(headers, null, 2),
    '',
    'Sample rows:',
    JSON.stringify(sampleObjects, null, 2),
    '',
    'Target row:',
    JSON.stringify(inputObject, null, 2)
  ].join('\n');
}

/**
 * Builds instructions for plain text cell responses.
 *
 * @param {boolean} useWebSearch Whether web tools are available.
 * @return {string} System instructions.
 */
function buildTextInstructions_(useWebSearch) {
  const base = 'Return only the concise value that should appear in the spreadsheet cell.';
  if (!useWebSearch) {
    return base;
  }

  return base + ' Use web_search for current or externally verifiable facts. Use fetch_url when a specific page is needed.';
}

/**
 * Builds instructions for auto-mode structured responses.
 *
 * @param {boolean} useWebSearch Whether web tools are available.
 * @return {string} System instructions.
 */
function buildAutoInstructions_(useWebSearch) {
  const base = [
    'You convert spreadsheet rows into a single structured spreadsheet result.',
    'Use the headers and sample rows to infer the intended schema and data meaning.',
    'Return JSON that exactly matches the provided schema.'
  ].join(' ');

  if (!useWebSearch) {
    return base;
  }

  return base + ' Use web_search for current or externally verifiable facts. Use fetch_url when a specific page is needed.';
}

/**
 * Converts a range or scalar into a two-dimensional array.
 *
 * @param {*} value Spreadsheet range value or scalar.
 * @return {Array<Array<*>>} Two-dimensional array.
 */
function normalizeRange_(value) {
  if (!Array.isArray(value)) {
    return [[value]];
  }

  if (value.length === 0) {
    return [];
  }

  if (!Array.isArray(value[0])) {
    return [value];
  }

  return value;
}

/**
 * Flattens a spreadsheet range into one row of values.
 *
 * @param {*} value Spreadsheet range value or scalar.
 * @return {Array<*>} Flattened row values.
 */
function flattenRange_(value) {
  const rows = normalizeRange_(value);
  if (rows.length === 0) {
    return [];
  }

  return rows.reduce(function (values, row) {
    return values.concat(row);
  }, []);
}

/**
 * Converts spreadsheet rows to objects keyed by headers.
 *
 * @param {Array<Array<*>>} rows Spreadsheet rows.
 * @param {string[]} headers Header names.
 * @return {Object[]} Row objects.
 */
function rowsToObjects_(rows, headers) {
  return rows
    .filter(function (row) {
      return row.some(function (value) {
        return String(value || '').trim() !== '';
      });
    })
    .map(function (row) {
      return rowToObject_(row, headers);
    });
}

/**
 * Converts a spreadsheet row to an object keyed by headers.
 *
 * @param {Array<*>} row Spreadsheet row.
 * @param {string[]} headers Header names.
 * @return {Object} Row object.
 */
function rowToObject_(row, headers) {
  const object = {};
  headers.forEach(function (header, index) {
    object[header] = normalizeCellValue_(row[index]);
  });
  return object;
}

/**
 * Normalizes one spreadsheet cell for prompt JSON.
 *
 * @param {*} value Cell value.
 * @return {string|number|boolean|null} Normalized value.
 */
function normalizeCellValue_(value) {
  if (value === undefined || value === null || value === '') {
    return null;
  }

  if (Object.prototype.toString.call(value) === '[object Date]') {
    return value.toISOString();
  }

  return value;
}

/**
 * Reads a script property.
 *
 * @param {string} name Property name.
 * @return {string} Property value.
 */
function getScriptProperty_(name) {
  return PropertiesService.getScriptProperties().getProperty(name) || '';
}

/**
 * Builds a short cache key for a request payload.
 *
 * @param {Object} payload Agent API request payload.
 * @return {string} Cache key.
 */
function buildCacheKey_(payload) {
  const bytes = Utilities.computeDigest(
    Utilities.DigestAlgorithm.SHA_256,
    JSON.stringify(payload),
    Utilities.Charset.UTF_8
  );
  return 'perplexity:' + Utilities.base64EncodeWebSafe(bytes).replace(/=+$/, '');
}
