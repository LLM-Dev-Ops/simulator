'use strict';

const Ajv = require('ajv');

const whatIfSchema = require('./what-if.schema.json');
const scenarioSchema = require('./scenario.schema.json');

const ajv = new Ajv({ allErrors: true, useDefaults: true });

const validateWhatIfBody = ajv.compile(whatIfSchema);
const validateScenarioBody = ajv.compile(scenarioSchema);

function formatErrors(errors) {
  if (!errors) return [];
  return errors.map((e) => ({
    field: e.instancePath || '/',
    message: e.message,
    params: e.params,
  }));
}

function validateWhatIfRequest(body) {
  if (!body || typeof body !== 'object') {
    return { valid: false, errors: [{ field: '/', message: 'Request body must be a JSON object' }] };
  }
  const valid = validateWhatIfBody(body);
  return { valid, errors: valid ? [] : formatErrors(validateWhatIfBody.errors) };
}

function validateScenarioRequest(body) {
  if (!body || typeof body !== 'object') {
    return { valid: false, errors: [{ field: '/', message: 'Request body must be a JSON object' }] };
  }
  const valid = validateScenarioBody(body);
  return { valid, errors: valid ? [] : formatErrors(validateScenarioBody.errors) };
}

module.exports = { validateWhatIfRequest, validateScenarioRequest };
